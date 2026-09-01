from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Union
from uuid import uuid4

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from ..utils.utility import useful_functions
from ..utils.reporting import emit_alert, emit_detail


@dataclass(frozen=True)
class _ImageTarget:
    galaxy: str
    observatory: str
    band: str
    data: Any
    header: fits.Header
    error: Any
    source_path: Path


@dataclass(frozen=True)
class _StagedTarget:
    target: _ImageTarget
    image_path: Path
    error_path: Path | None


@dataclass(frozen=True)
class _RegisteredTarget:
    target: _ImageTarget
    data: np.ndarray
    header: fits.Header
    error: np.ndarray | None


class Register:
    _WCS_KEY_PATTERNS = tuple(
        re.compile(pattern)
        for pattern in (
            # FITS-WCS core keywords, including alternate WCS suffixes A-Z.
            r"^(?:WCSAXES|CRPIX\d+|CDELT\d+|CUNIT\d+|CTYPE\d+|CRVAL\d+|CROTA\d+|"
            r"CD\d+_\d+|PC\d+_\d+|PV\d+_\d+|PS\d+_\d+|RADESYS|RADECSYS|"
            r"EQUINOX|LONPOLE|LATPOLE|WCSNAME)[A-Z]?$",
            # Deprecated projection parameters used by older FITS writers.
            r"^PROJP\d+[A-Z]?$",
            # SIP polynomial distortion keywords.
            r"^(?:A|B|AP|BP)_(?:ORDER|\d+_\d+)[A-Z]?$",
            # Distortion lookup-table descriptors and their parameter records.
            r"^(?:CPDIS|CQDIS|D2IMDIS|DET2IM|CPERR|CQERR|D2IMERR)\d+[A-Z]?$",
            r"^(?:DP|DQ|D2IM)\d+(?:[A-Z]|\..+)?$",
        )
    )
    _WCS_ROUNDTRIP_TOLERANCE_PIXELS = 1.0e-4

    @classmethod
    def register_image_set(cls, image_set, reporter=None) -> dict[str, Path]:
        """Register every image currently stored in a ``GalaxyImageSet``.

        Each invocation uses an isolated workspace and invokes SWarp exactly once
        per galaxy. Existing files in sibling or legacy SWarp directories are never
        scanned, selected, renamed, or deleted.

        Returns
        -------
        dict[str, pathlib.Path]
            Retained audit directory for each galaxy. Successful runs retain only
            the input manifest and SWarp log; failed runs retain the full workspace.
        """
        targets_by_galaxy = cls._collect_targets(image_set)
        if not targets_by_galaxy:
            raise ValueError("Cannot run SWarp registration on an empty GalaxyImageSet")

        run_id = cls._new_run_id()
        run_directories: dict[str, Path] = {}

        for galaxy, targets in targets_by_galaxy.items():
            if reporter is not None:
                reporter.set_target(
                    f"{galaxy} (SWarp: {len(targets)} images)"
                )
            run_dir = cls._create_run_directory(targets[0].source_path, run_id, galaxy)
            run_directories[galaxy] = run_dir
            input_dir = run_dir / "inputs"
            resample_dir = run_dir / "resampled"
            manifest_path = run_dir / f"{cls._safe_component(galaxy)}.list"
            log_path = run_dir / "swarp.log"

            input_dir.mkdir()
            resample_dir.mkdir()

            try:
                staged_targets = cls._stage_targets(targets, input_dir)
                cls._write_manifest(staged_targets, manifest_path, run_dir)
                cls.swarp(
                    input_list=manifest_path,
                    output=run_dir / "coadd.fits",
                    dump_dir=run_dir,
                    resample_dir=resample_dir,
                    log_file=log_path,
                )
                registered_targets = cls._load_registered_targets(
                    staged_targets,
                    resample_dir,
                    log_path,
                )
                cls._apply_registered_targets(
                    image_set,
                    registered_targets,
                    reporter=reporter,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"SWarp registration failed for {galaxy}. "
                    f"Workspace retained at {run_dir}: {exc}"
                ) from exc
            else:
                cls._cleanup_successful_run(run_dir, manifest_path, log_path)

        return run_directories

    @classmethod
    def _collect_targets(cls, image_set) -> dict[str, list[_ImageTarget]]:
        data_tree = getattr(image_set, "data", None)
        header_tree = getattr(image_set, "header", None)
        file_tree = getattr(image_set, "files", None)
        error_tree = getattr(image_set, "error", {})

        if not isinstance(data_tree, dict):
            raise TypeError("image_set.data must be a nested dictionary")
        if not isinstance(header_tree, dict):
            raise TypeError("image_set.header must be a nested dictionary")
        if not isinstance(file_tree, dict):
            raise TypeError("image_set.files must be a nested dictionary")

        targets_by_galaxy: dict[str, list[_ImageTarget]] = {}
        for galaxy, observatories in data_tree.items():
            for observatory, bands in observatories.items():
                for band, data in bands.items():
                    try:
                        header = header_tree[galaxy][observatory][band]
                        source_path = Path(file_tree[galaxy][observatory][band])
                    except KeyError as exc:
                        raise ValueError(
                            "GalaxyImageSet data, header, and file mappings are out of sync "
                            f"at {galaxy}/{observatory}/{band}"
                        ) from exc

                    try:
                        error = error_tree[galaxy][observatory][band]
                    except (KeyError, TypeError):
                        error = None

                    targets_by_galaxy.setdefault(galaxy, []).append(
                        _ImageTarget(
                            galaxy=galaxy,
                            observatory=observatory,
                            band=band,
                            data=data,
                            header=header,
                            error=error,
                            source_path=source_path,
                        )
                    )

        return targets_by_galaxy

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return f"{timestamp}-{uuid4().hex[:8]}"

    @classmethod
    def _create_run_directory(cls, source_path: Path, run_id: str, galaxy: str) -> Path:
        run_dir = (
            source_path.parent
            / "SWarp"
            / "runs"
            / run_id
            / cls._safe_component(galaxy)
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @staticmethod
    def _safe_component(value: Any) -> str:
        component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
        return component or "unknown"

    @classmethod
    def _stage_targets(
        cls,
        targets: list[_ImageTarget],
        input_dir: Path,
    ) -> list[_StagedTarget]:
        staged_targets = []
        for index, target in enumerate(targets):
            source_stem = cls._safe_component(target.source_path.stem)
            observatory = cls._safe_component(target.observatory)
            band = cls._safe_component(target.band)
            stage_stem = f"{index:04d}-{observatory}-{band}-{source_stem}"
            image_path = input_dir / f"{stage_stem}.fits"

            image_data = np.asarray(target.data)
            fits.writeto(image_path, image_data, target.header, overwrite=False)

            error_path = None
            if target.error is not None:
                error_data = np.asarray(target.error)
                if error_data.shape != image_data.shape:
                    raise ValueError(
                        f"Error image shape {error_data.shape} does not match science image "
                        f"shape {image_data.shape} for "
                        f"{target.galaxy}/{target.observatory}/{target.band}"
                    )
                error_path = input_dir / f"{stage_stem}_err.fits"
                fits.writeto(error_path, error_data, target.header, overwrite=False)

            staged_targets.append(
                _StagedTarget(
                    target=target,
                    image_path=image_path,
                    error_path=error_path,
                )
            )

        return staged_targets

    @staticmethod
    def _write_manifest(
        staged_targets: list[_StagedTarget],
        manifest_path: Path,
        run_dir: Path,
    ) -> None:
        manifest_lines = []
        for staged in staged_targets:
            manifest_lines.append(staged.image_path.relative_to(run_dir).as_posix())
            if staged.error_path is not None:
                manifest_lines.append(staged.error_path.relative_to(run_dir).as_posix())
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    @classmethod
    def _load_registered_targets(
        cls,
        staged_targets: list[_StagedTarget],
        resample_dir: Path,
        log_path: Path,
    ) -> list[_RegisteredTarget]:
        expected_outputs: list[Path] = []
        output_pairs: list[tuple[Path, Path | None]] = []

        for staged in staged_targets:
            image_output = cls._resampled_path(staged.image_path, resample_dir)
            error_output = None
            if staged.error_path is not None:
                error_output = cls._resampled_path(staged.error_path, resample_dir)
            output_pairs.append((image_output, error_output))
            expected_outputs.append(image_output)
            if error_output is not None:
                expected_outputs.append(error_output)

        missing_outputs = [path for path in expected_outputs if not path.is_file()]
        if missing_outputs:
            missing_names = ", ".join(path.name for path in missing_outputs)
            raise RuntimeError(
                f"SWarp did not produce expected resampled files: {missing_names}. "
                f"See log: {log_path}"
            )

        registered_targets = []
        for staged, (image_output, error_output) in zip(staged_targets, output_pairs):
            registered_data = fits.getdata(image_output).astype(np.float32)
            registered_header = fits.getheader(image_output)
            updated_header = cls._merge_registered_header(
                staged.target.header,
                registered_header,
                shape=registered_data.shape,
                context=(
                    "SWarp registration for "
                    f"{staged.target.galaxy}/{staged.target.observatory}/{staged.target.band}"
                ),
            )

            registered_error = None
            if error_output is not None:
                registered_error = fits.getdata(error_output).astype(np.float32)
                if registered_error.shape != registered_data.shape:
                    raise RuntimeError(
                        "Registered science and error images have different shapes for "
                        f"{staged.target.galaxy}/{staged.target.observatory}/{staged.target.band}"
                    )

            registered_targets.append(
                _RegisteredTarget(
                    target=staged.target,
                    data=registered_data,
                    header=updated_header,
                    error=registered_error,
                )
            )

        return registered_targets

    @staticmethod
    def _resampled_path(input_path: Path, resample_dir: Path) -> Path:
        return resample_dir / f"{input_path.stem}.resamp{input_path.suffix}"

    @classmethod
    def _merge_registered_header(
        cls,
        original_header: fits.Header,
        registered_header: fits.Header,
        *,
        shape=None,
        context="SWarp registration",
    ) -> fits.Header:
        if shape is None:
            shape = (registered_header.get("NAXIS2"), registered_header.get("NAXIS1"))
        return cls._replace_celestial_wcs(
            original_header,
            registered_header,
            shape=shape,
            context=context,
        )

    @classmethod
    def _replace_celestial_wcs(
        cls,
        base_header: fits.Header,
        authoritative_header: fits.Header,
        *,
        shape,
        context: str,
        target_coord: SkyCoord | None = None,
    ) -> fits.Header:
        """Replace all celestial WCS cards while preserving non-WCS metadata."""
        normalized_shape = cls._validate_2d_shape(shape, context)
        canonical_wcs = cls._canonical_celestial_header(authoritative_header, context)

        base_metadata = cls._without_wcs_cards(base_header)
        authoritative_metadata = cls._without_wcs_cards(authoritative_header)
        updated_header = useful_functions.update_header(base_metadata, authoritative_metadata)
        updated_header.update(canonical_wcs)
        for key in list(updated_header.keys()):
            if re.fullmatch(r"NAXIS\d+", str(key).strip().upper()):
                while key in updated_header:
                    del updated_header[key]
        updated_header["NAXIS"] = 2
        updated_header["NAXIS1"] = normalized_shape[1]
        updated_header["NAXIS2"] = normalized_shape[0]

        cls._validate_celestial_wcs(
            updated_header,
            normalized_shape,
            context=context,
            target_coord=target_coord,
        )
        return updated_header

    @classmethod
    def _without_wcs_cards(cls, header: fits.Header) -> fits.Header:
        cleaned = header.copy()
        for key in list(cleaned.keys()):
            if cls._is_wcs_keyword(key):
                while key in cleaned:
                    del cleaned[key]
        return cleaned

    @classmethod
    def _is_wcs_keyword(cls, keyword: str) -> bool:
        normalized = str(keyword).strip().upper()
        return any(pattern.fullmatch(normalized) for pattern in cls._WCS_KEY_PATTERNS)

    @staticmethod
    def _validate_2d_shape(shape, context: str) -> tuple[int, int]:
        try:
            normalized = tuple(int(value) for value in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: image shape {shape!r} is invalid") from exc
        if len(normalized) != 2 or any(value <= 0 for value in normalized):
            raise ValueError(
                f"{context}: a positive 2-D image shape is required, got {normalized}"
            )
        return normalized

    @staticmethod
    def _canonical_celestial_header(header: fits.Header, context: str) -> fits.Header:
        try:
            celestial = WCS(header, relax=True).celestial
        except Exception as exc:
            raise ValueError(f"{context}: celestial WCS could not be parsed ({exc})") from exc
        if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
            raise ValueError(
                f"{context}: exactly two celestial WCS axes are required; "
                f"got pixel_n_dim={celestial.pixel_n_dim}, world_n_dim={celestial.world_n_dim}"
            )
        try:
            return celestial.to_header(relax=True)
        except Exception as exc:
            raise ValueError(f"{context}: celestial WCS could not be serialized ({exc})") from exc

    @classmethod
    def _validate_celestial_wcs(
        cls,
        header: fits.Header,
        shape,
        *,
        context: str,
        target_coord: SkyCoord | None = None,
    ) -> None:
        height, width = cls._validate_2d_shape(shape, context)
        try:
            wcs = WCS(header, relax=True).celestial
            scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float)
        except Exception as exc:
            raise ValueError(f"{context}: normalized celestial WCS is invalid ({exc})") from exc

        if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
            raise ValueError(f"{context}: normalized WCS is not two-dimensional and celestial")
        if scales.shape != (2,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError(f"{context}: invalid celestial pixel scales {scales.tolist()}")

        if target_coord is None:
            test_x = (width - 1.0) / 2.0
            test_y = (height - 1.0) / 2.0
        else:
            if not target_coord.isscalar:
                raise ValueError(f"{context}: target coordinate must be scalar")
            try:
                test_x, test_y = wcs.world_to_pixel(target_coord)
            except Exception as exc:
                raise ValueError(
                    f"{context}: target coordinate could not be projected ({exc})"
                ) from exc
            if not (
                np.isfinite(test_x)
                and np.isfinite(test_y)
                and 0.0 <= float(test_x) < float(width)
                and 0.0 <= float(test_y) < float(height)
            ):
                raise ValueError(
                    f"{context}: target coordinate projects outside image shape "
                    f"{(height, width)} at ({test_x}, {test_y})"
                )

        try:
            world = wcs.pixel_to_world(test_x, test_y)
            roundtrip_x, roundtrip_y = wcs.world_to_pixel(world)
            error = float(np.hypot(roundtrip_x - test_x, roundtrip_y - test_y))
        except Exception as exc:
            raise ValueError(f"{context}: WCS round-trip failed ({exc})") from exc
        if not np.isfinite(error) or error > cls._WCS_ROUNDTRIP_TOLERANCE_PIXELS:
            raise ValueError(
                f"{context}: WCS round-trip error {error} pixels exceeds "
                f"{cls._WCS_ROUNDTRIP_TOLERANCE_PIXELS}"
            )

    @staticmethod
    def _apply_registered_targets(
        image_set,
        registered_targets: list[_RegisteredTarget],
        reporter=None,
    ) -> None:
        for registered in registered_targets:
            target = registered.target
            image_set.update_data(
                registered.data,
                target.galaxy,
                target.observatory,
                target.band,
            )
            image_set.update_header(
                registered.header,
                target.galaxy,
                target.observatory,
                target.band,
            )
            if registered.error is not None:
                image_set.update_error(
                    registered.error,
                    target.galaxy,
                    target.observatory,
                    target.band,
                )
            if reporter is not None:
                reporter.advance_target(
                    f"{target.galaxy}/{target.observatory}/{target.band}"
                )

    @staticmethod
    def _cleanup_successful_run(run_dir: Path, manifest_path: Path, log_path: Path) -> None:
        retained_paths = {manifest_path, log_path}
        try:
            for path in run_dir.iterdir():
                if path in retained_paths:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        except OSError as exc:
            warnings.warn(
                f"SWarp registration succeeded, but workspace cleanup failed at "
                f"{run_dir}: {exc}"
            )

    @classmethod
    def swarp(
        cls,
        input_list,
        output=None,
        dump_dir=None,
        resample_dir=None,
        log_file=None,
    ):
        """Execute SWarp with either an input manifest or a list of filenames."""
        if isinstance(input_list, (str, Path)):
            manifest = str(input_list)
            input_argument = manifest if manifest.startswith("@") else f"@{manifest}"
            input_parent = Path(manifest.removeprefix("@")).parent
        elif isinstance(input_list, list):
            if not input_list:
                raise ValueError("SWarp input list cannot be empty")
            input_argument = ",".join(str(path) for path in input_list)
            input_parent = Path.cwd()
        else:
            raise ValueError("Input must be a list, string, or pathlib.Path")

        dump_dir = Path(dump_dir) if dump_dir is not None else input_parent / "tmp_swarp"
        resample_dir = Path(resample_dir) if resample_dir is not None else dump_dir / "resamp"
        output = Path(output) if output is not None else dump_dir / "coadd.fits"
        log_file = Path(log_file) if log_file is not None else dump_dir / "swarp.log"

        dump_dir.mkdir(parents=True, exist_ok=True)
        resample_dir.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        config_file_resource = resources.files("Spec7DT.reference.configs").joinpath(
            "default.swarp"
        )
        with resources.as_file(config_file_resource) as config_path:
            command = [
                "swarp",
                input_argument,
                "-c",
                str(config_path),
                "-IMAGEOUT_NAME",
                str(output),
                "-RESAMPLE_DIR",
                str(resample_dir),
            ]

            command_text = shlex.join(command)
            with log_file.open("w", encoding="utf-8") as log:
                log.write(command_text + "\n\n\n")
                log.flush()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=str(dump_dir),
                        shell=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                except OSError as exc:
                    log.write(f"Failed to start SWarp: {exc}\n")
                    raise RuntimeError(f"Unable to start SWarp. See log: {log_file}") from exc

                stdout = process.stdout
                try:
                    for line in stdout or ():
                        log.write(line)
                        log.flush()
                finally:
                    close_stdout = getattr(stdout, "close", None)
                    if close_stdout is not None:
                        close_stdout()

                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(
                        f"SWarp failed with return code {process.returncode}. "
                        f"See log: {log_file}"
                    )

        return command_text

    @classmethod
    def pretrim_for_convolution(
        cls,
        image_data,
        header,
        error_data,
        galaxy_name,
        observatory,
        band,
        image_set,
        pretrim_fov,
        kernel_truncate=4.0,
        metadata_resolver=None,
    ):
        """Crop to a sky FoV plus PSF halo, padding unavailable pixels with NaN."""
        from .PSF import PointSpreadFunction

        target_name = f"{galaxy_name}/{observatory}/{band}"
        original_shape = np.shape(image_data)
        if len(original_shape) != 2:
            raise ValueError(f"Pre-convolution trim requires a 2-D image for {target_name}")
        if error_data is not None and np.shape(error_data) != original_shape:
            raise ValueError(
                f"STD error map shape {np.shape(error_data)} does not match science image "
                f"shape {original_shape} for {target_name}"
            )

        center_coord = useful_functions.get_sky_loc(
            galaxy_name,
            header=header,
            metadata_resolver=metadata_resolver,
            required=True,
            image_shape=original_shape,
        )
        try:
            matrix = PointSpreadFunction._pixel_scale_matrix_arcsec(header)
        except ValueError as exc:
            raise ValueError(f"Invalid WCS for pre-convolution trim of {target_name}: {exc}") from exc
        axis_scales = np.sqrt(np.sum(matrix ** 2, axis=0))

        sigma_extra = PointSpreadFunction.extra_sigma_arcsec(
            image_set,
            galaxy_name,
            observatory,
            band,
        )
        halo_arcsec = 0.0 if sigma_extra is None else kernel_truncate * sigma_extra
        requested_size = tuple(
            axis_size + 2.0 * halo_arcsec * u.arcsec
            for axis_size in pretrim_fov
        )

        try:
            trim_image, trim_header, trim_error, coverage_info = cls._trim_sky_with_info(
                image=image_data,
                header=header,
                error=error_data,
                skycoord=center_coord,
                size=requested_size,
                mode="partial",
                fill_value=np.nan,
                context=f"pre-convolution trim for {target_name}",
            )
        except NoOverlapError as exc:
            requested_text = tuple(round(value.to_value(u.arcsec), 6) for value in requested_size)
            scale_text = tuple(round(float(value), 6) for value in axis_scales[::-1])
            raise ValueError(
                f"Pre-convolution FoV does not overlap the image for {target_name}: "
                f"requested(ny,nx)={requested_text} arcsec, input_shape={original_shape}, "
                f"pixel_scale(ny,nx)={scale_text} arcsec/pixel"
            ) from exc

        if coverage_info["coverage"] == "partial":
            emit_alert(
                f"Pre-convolution FoV is partially covered for {target_name}: "
                f"requested_shape={coverage_info['requested_shape']}, "
                f"overlap_shape={coverage_info['overlap_shape']}, "
                f"coverage={coverage_info['coverage_fraction']:.3f}; "
                "padding unavailable science and error pixels with NaN.",
                context=target_name,
                dedupe_key="psf.preconvolution.partial_fov",
            )

        image_set.update_data(trim_image, galaxy_name, observatory, band)
        image_set.update_header(trim_header, galaxy_name, observatory, band)
        if error_data is not None:
            image_set.update_error(trim_error, galaxy_name, observatory, band)

        sigma_text = "invalid" if sigma_extra is None else f"{sigma_extra:.6g}"
        emit_detail(
            f"PSF pretrim {target_name}: shape={original_shape}->{trim_image.shape}, "
            f"core_fov={tuple(value.to_value(u.arcsec) for value in pretrim_fov)} arcsec, "
            f"sigma_extra={sigma_text} arcsec, halo={halo_arcsec:.6g} arcsec, "
            f"coverage={coverage_info['coverage']}:{coverage_info['coverage_fraction']:.3f}"
        )

    @classmethod
    def trim(
        cls,
        image_data,
        header,
        error_data,
        galaxy_name,
        observatory,
        band,
        image_set,
        trim_size,
        metadata_resolver=None,
    ):
        center_coord = useful_functions.get_sky_loc(
            galaxy_name,
            header=header,
            metadata_resolver=metadata_resolver,
            required=True,
            image_shape=image_data.shape,
        )

        target_name = f"{galaxy_name}/{observatory}/{band}"
        try:
            trim_image, trim_header, trim_error, _ = cls._trim_sky_with_info(
                image=image_data,
                header=header,
                error=error_data,
                skycoord=center_coord,
                size=trim_size,
                mode="strict",
                context=f"final trim for {target_name}",
            )
        except (PartialOverlapError, NoOverlapError) as exc:
            raise ValueError(
                f"Final trim size {trim_size} is not fully covered for {target_name}; "
                f"registered image shape is {np.shape(image_data)}"
            ) from exc

        image_set.update_data(trim_image, galaxy_name, observatory, band)
        image_set.update_header(trim_header, galaxy_name, observatory, band)
        if error_data is not None:
            image_set.update_error(trim_error, galaxy_name, observatory, band)

    @classmethod
    def trim_sky(
        cls,
        image,
        header,
        error,
        skycoord: Union[tuple, SkyCoord],
        size: tuple,
        mode: str = "strict",
    ):
        cut_image, new_header, cut_error, _ = cls._trim_sky_with_info(
            image=image,
            header=header,
            error=error,
            skycoord=skycoord,
            size=size,
            mode=mode,
        )
        return cut_image, new_header, cut_error

    @classmethod
    def _trim_sky_with_info(
        cls,
        image,
        header,
        error,
        skycoord: Union[tuple, SkyCoord],
        size: tuple,
        mode: str = "strict",
        fill_value=0.0,
        context: str = "sky cutout",
    ):
        wcs = WCS(header)

        skycoord_obj = (
            SkyCoord(ra=skycoord[0] * u.deg, dec=skycoord[1] * u.deg)
            if isinstance(skycoord, tuple)
            else skycoord
        )
        pixel_position = skycoord_obj.to_pixel(wcs=wcs)

        image_array = np.asarray(image)
        error_array = None if error is None else np.asarray(error)
        if not np.isfinite(fill_value):
            if not np.issubdtype(image_array.dtype, np.inexact):
                image_array = image_array.astype(np.float32)
            if error_array is not None and not np.issubdtype(error_array.dtype, np.inexact):
                error_array = error_array.astype(np.float32)

        cut = Cutout2D(
            image_array,
            pixel_position,
            size,
            wcs=wcs,
            mode=mode,
            fill_value=fill_value,
            copy=True,
        )
        cut_error = (
            Cutout2D(
                error_array,
                pixel_position,
                size,
                wcs=wcs,
                mode=mode,
                fill_value=fill_value,
                copy=True,
            )
            if error_array is not None
            else None
        )

        requested_shape = tuple(int(value) for value in cut.shape_input)
        overlap_shape = tuple(
            int(axis_slice.stop - axis_slice.start)
            for axis_slice in cut.slices_original
        )
        requested_pixels = int(np.prod(requested_shape))
        overlap_pixels = int(np.prod(overlap_shape))
        coverage_fraction = (
            float(overlap_pixels / requested_pixels)
            if requested_pixels > 0
            else 0.0
        )
        coverage_info = {
            "coverage": "full" if overlap_shape == requested_shape else "partial",
            "coverage_fraction": coverage_fraction,
            "requested_shape": requested_shape,
            "overlap_shape": overlap_shape,
        }

        new_header = cls._replace_celestial_wcs(
            header,
            cut.wcs.to_header(relax=True),
            shape=cut.data.shape,
            context=context,
            target_coord=skycoord_obj,
        )

        cut_error_data = None if cut_error is None else np.asarray(cut_error.data, order="C")
        return np.asarray(cut.data, order="C"), new_header, cut_error_data, coverage_info
