from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.table import Table


CHECKPOINT_SCHEMA_VERSION = 3


def _package_version():
    try:
        return version("Spec7DT")
    except PackageNotFoundError:
        return "0.14.0"


def _safe_component(value):
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return component or "unknown"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _new_run_id():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def to_jsonable(value):
    """Convert supported pipeline settings and state to safe JSON values."""
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return {"__type__": "float", "value": str(value)}
        return value
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, u.Quantity):
        return {
            "__type__": "quantity",
            "value": to_jsonable(np.asarray(value.value).tolist()),
            "unit": str(value.unit),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "columns": [str(column) for column in value.columns],
            "records": [
                [to_jsonable(item) for item in row]
                for row in value.itertuples(index=False, name=None)
            ],
        }
    if isinstance(value, Table):
        return {
            "__type__": "astropy_table",
            "table": to_jsonable(value.to_pandas()),
        }
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "value": to_jsonable(value.tolist())}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"Checkpoint value of type {type(value).__name__} is not JSON serializable")


def from_jsonable(value):
    if isinstance(value, list):
        return [from_jsonable(item) for item in value]
    if not isinstance(value, dict):
        return value

    marker = value.get("__type__")
    if marker == "float":
        return float(value["value"])
    if marker == "path":
        return Path(value["value"])
    if marker == "datetime":
        return datetime.fromisoformat(value["value"])
    if marker == "quantity":
        return np.asarray(from_jsonable(value["value"])) * u.Unit(value["unit"])
    if marker == "dataframe":
        records = [
            [from_jsonable(item) for item in row]
            for row in value.get("records", [])
        ]
        return pd.DataFrame(records, columns=value.get("columns", []))
    if marker == "astropy_table":
        return Table.from_pandas(from_jsonable(value["table"]))
    if marker == "ndarray":
        return np.asarray(from_jsonable(value["value"]))
    return {key: from_jsonable(item) for key, item in value.items()}


def _read_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_manifest_atomic(path, manifest):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(to_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_path(resume_from):
    path = Path(resume_from).expanduser().resolve()
    if path.is_file():
        if path.name != "manifest.json":
            raise ValueError(f"Checkpoint file must be named manifest.json: {path}")
        return path
    direct = path / "manifest.json"
    if direct.is_file():
        return direct
    nested = sorted(path.glob("*/manifest.json")) if path.is_dir() else []
    if nested:
        return nested[0]
    raise FileNotFoundError(f"No checkpoint manifest found at {path}")


def _copy_tree(value):
    """Copy nested dictionaries/lists without copying large image arrays."""
    if isinstance(value, dict):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_tree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rewrite_paths(value, replacements):
    """Rewrite absolute paths below generated artifact roots, preserving types."""

    if isinstance(value, dict):
        return {key: _rewrite_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_paths(item, replacements) for item in value)
    if not isinstance(value, (str, Path)):
        return value
    original = Path(value).expanduser()
    if not original.is_absolute():
        return value
    resolved = original.resolve()
    for source, destination in replacements:
        try:
            relative = resolved.relative_to(source)
        except ValueError:
            continue
        rewritten = destination / relative
        return rewritten if isinstance(value, Path) else str(rewritten)
    return value


def _checkpoint_spherex_psf_builds(image_set, root, settings):
    builds = tuple(getattr(image_set, "_spherex_psf_builds", ()))
    if not builds:
        return _copy_tree(settings), []
    destination_root = Path(root) / "artifacts" / "psf_synthesis"
    destination_root.mkdir(parents=True, exist_ok=False)
    replacements = []
    records = []
    for build in builds:
        source = Path(build.manifest_path).expanduser().resolve().parent
        build_digest = str(build.manifest.get("build_digest") or source.name)
        destination = destination_root / _safe_component(build_digest)
        shutil.copytree(source, destination)
        replacements.append((source, destination.resolve()))
        records.append({
            "build_digest": build_digest,
            "manifest": str((destination / Path(build.manifest_path).name).relative_to(root)),
            "exposure_snapshot": str(
                (destination / Path(build.exposure_snapshot_path).name).relative_to(root)
            ),
            "artifacts": [
                str((destination / Path(path).name).relative_to(root))
                for path in build.artifact_paths
            ],
            "qc_digest": sha256(
                json.dumps(to_jsonable(build.qc), sort_keys=True).encode("utf-8")
            ).hexdigest(),
        })
    return _rewrite_paths(_copy_tree(settings), replacements), records


def _checkpoint_gaia_state(image_set, galaxy):
    reference = getattr(image_set, "_gaia_mask_reference_cache", {}).get(galaxy)
    if reference is None:
        return None
    if not isinstance(reference, dict):
        return reference
    # Masking after a restart only consumes ``mask_sources``. Keep the compact
    # audit fields too, but do not duplicate the reference image in JSON.
    excluded = {"reference_image", "reference_header", "detections"}
    return {
        key: _copy_tree(value)
        for key, value in reference.items()
        if key not in excluded
    }


@dataclass
class CheckpointResume:
    image_set: object
    settings: dict
    completed_step_ids: tuple[str, ...]
    manager: "CheckpointManager"


class CheckpointManager:
    """Write restartable, per-galaxy pipeline checkpoints."""

    def __init__(self, image_set, manifests, roots):
        self.image_set = image_set
        self.manifests = manifests
        self.roots = {str(key): Path(value) for key, value in roots.items()}

    @property
    def output_dirs(self):
        return dict(self.roots)

    @classmethod
    def create(cls, image_set, *, settings, step_plan, checkpoint_dir=None):
        galaxies = sorted(str(galaxy) for galaxy in image_set.data)
        if not galaxies:
            raise ValueError("Cannot create checkpoints for an empty GalaxyImageSet")

        run_id = _new_run_id()
        roots = {}
        automatic_candidates = {}
        explicit_base = None if checkpoint_dir is None else Path(checkpoint_dir).expanduser().resolve()
        for galaxy in galaxies:
            if explicit_base is not None:
                root = explicit_base / run_id / _safe_component(galaxy)
            else:
                paths = []
                for bands in image_set.files.get(galaxy, {}).values():
                    paths.extend(Path(path).expanduser().resolve() for path in bands.values())
                if not paths:
                    raise ValueError(
                        f"No source path is available for {galaxy}; set checkpoint_dir explicitly"
                    )
                representative = sorted(paths, key=lambda item: str(item))[0]
                root = representative.parent / "Spec7DT" / "runs" / run_id
            automatic_candidates.setdefault(str(root), []).append(galaxy)
            roots[galaxy] = root

        # Galaxies can share one input directory. Preserve the compact documented
        # layout when roots are unique, and add a galaxy component only on collision.
        for shared_root, members in automatic_candidates.items():
            if len(members) > 1:
                for galaxy in members:
                    roots[galaxy] = Path(shared_root) / _safe_component(galaxy)

        for root in roots.values():
            root.mkdir(parents=True, exist_ok=False)
            (root / "checkpoints").mkdir()
            (root / "artifacts" / "psf").mkdir(parents=True)

        checkpoint_settings, synthesis_records = _checkpoint_spherex_psf_builds(
            image_set,
            roots[galaxies[0]],
            settings,
        )

        members = {galaxy: str(root.resolve()) for galaxy, root in roots.items()}
        manifests = {}
        created = _utc_now()
        package_version = _package_version()
        for galaxy in galaxies:
            inputs = []
            for observatory, bands in image_set.data[galaxy].items():
                for band in bands:
                    source_path = Path(image_set.files[galaxy][observatory][band]).expanduser().resolve()
                    error_path = source_path.with_name(source_path.stem + "_err").with_suffix(source_path.suffix)
                    inputs.append({
                        "observatory": str(observatory),
                        "band": str(band),
                        "science": str(source_path),
                        "error": str(error_path) if error_path.is_file() else None,
                        "product_metadata": (
                            image_set.product_metadata[galaxy][observatory][band].to_dict()
                            if band in image_set.product_metadata.get(galaxy, {}).get(observatory, {})
                            else None
                        ),
                    })
            manifest = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "spec7dt_version": package_version,
                "run_id": run_id,
                "created_at": created,
                "updated_at": created,
                "galaxy": galaxy,
                "members": members,
                "settings": checkpoint_settings,
                "step_plan": step_plan,
                "completed_step_ids": [],
                "latest_checkpoint": None,
                "inputs": inputs,
                "state": {},
                "products": [],
                "psf_synthesis": synthesis_records,
            }
            manifests[galaxy] = manifest
            _write_manifest_atomic(roots[galaxy] / "manifest.json", manifest)

        manager = cls(image_set, manifests, roots)
        manager._publish_output_dirs()
        return manager

    @classmethod
    def resume(cls, resume_from):
        first_path = _manifest_path(resume_from)
        first = _read_manifest(first_path)
        cls._validate_manifest(first, first_path)
        run_id = first["run_id"]

        manifests = {}
        roots = {}
        for galaxy, root_value in first.get("members", {}).items():
            root = Path(root_value).expanduser().resolve()
            path = root / "manifest.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint member for {galaxy} is missing its manifest: {path}"
                )
            manifest = _read_manifest(path)
            cls._validate_manifest(manifest, path)
            if manifest.get("run_id") != run_id:
                raise ValueError(f"Checkpoint run ID mismatch at {path}")
            manifests[str(galaxy)] = manifest
            roots[str(galaxy)] = root

        if not manifests:
            galaxy = str(first["galaxy"])
            manifests[galaxy] = first
            roots[galaxy] = first_path.parent

        completed_sets = {
            tuple(manifest.get("completed_step_ids", ()))
            for manifest in manifests.values()
        }
        if len(completed_sets) != 1:
            raise ValueError("Checkpoint member manifests disagree about completed pipeline steps")
        completed = next(iter(completed_sets))
        image_set = cls._restore_image_set(manifests, roots)
        manager = cls(image_set, manifests, roots)
        manager._publish_output_dirs()
        settings = from_jsonable(first.get("settings", {}))
        return CheckpointResume(image_set, settings, completed, manager)

    @staticmethod
    def _validate_manifest(manifest, path):
        schema = manifest.get("schema_version")
        if schema not in {1, 2, CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError(
                f"Unsupported checkpoint schema {schema!r} at {path}; "
                f"expected 1, 2, or {CHECKPOINT_SCHEMA_VERSION}"
            )
        saved_version = manifest.get("spec7dt_version")
        current_version = _package_version()
        if saved_version != current_version:
            raise ValueError(
                f"Checkpoint Spec7DT version {saved_version!r} does not match "
                f"the installed version {current_version!r}"
            )

    @classmethod
    def _restore_image_set(cls, manifests, roots):
        from .file_handler import GalaxyImageSet

        image_set = GalaxyImageSet(auto_ensure_filters=False)
        for galaxy, manifest in manifests.items():
            state = from_jsonable(manifest.get("state", {}))
            checkpoint = manifest.get("latest_checkpoint")
            records = checkpoint.get("files", []) if checkpoint else manifest.get("inputs", [])
            for record in records:
                observatory = str(record["observatory"])
                band = str(record["band"])
                science_value = record["science"]
                science_path = Path(science_value)
                if checkpoint and not science_path.is_absolute():
                    science_path = roots[galaxy] / science_path
                if not science_path.is_file():
                    raise FileNotFoundError(f"Checkpoint science FITS is missing: {science_path}")
                data, header = GalaxyImageSet._read_fits_image(science_path)

                error = None
                error_value = record.get("error")
                if error_value:
                    error_path = Path(error_value)
                    if checkpoint and not error_path.is_absolute():
                        error_path = roots[galaxy] / error_path
                    if not error_path.is_file():
                        raise FileNotFoundError(f"Checkpoint error FITS is missing: {error_path}")
                    error, _ = GalaxyImageSet._read_fits_image(error_path)

                for tree in (
                    image_set._data,
                    image_set._header,
                    image_set._error,
                    image_set._error_source,
                    image_set._psf,
                    image_set._cutout_shape,
                    image_set._files,
                    image_set._product_metadata,
                ):
                    tree.setdefault(galaxy, {}).setdefault(observatory, {})
                image_set._data[galaxy][observatory][band] = data
                image_set._header[galaxy][observatory][band] = header
                image_set._error[galaxy][observatory][band] = error
                image_set._files[galaxy][observatory][band] = str(science_path)
                from ..psf.models import ImageProductMetadata

                image_set._product_metadata[galaxy][observatory][band] = (
                    ImageProductMetadata.from_value(
                        record.get("product_metadata"),
                        facility=observatory,
                        channel=band,
                        source_path=str(science_path),
                    )
                )

                source = (
                    state.get("error_source", {})
                    .get(observatory, {})
                    .get(band, "missing" if error is None else "provided")
                )
                image_set._error_source[galaxy][observatory][band] = source
                image_set._psf[galaxy][observatory][band] = (
                    state.get("psf", {}).get(observatory, {}).get(band, "")
                )
                image_set._cutout_shape[galaxy][observatory][band] = (
                    state.get("cutout_shape", {}).get(observatory, {}).get(band, set())
                )

            safe_artifacts = state.get("safe_psf_artifacts", {})
            image_set._safe_psf_failures.update(state.get("safe_psf_failures", {}))
            if safe_artifacts:
                from ..reduction.PSF_safe import read_psf_artifact

                for observatory, bands in safe_artifacts.items():
                    for band, artifact_record in bands.items():
                        if isinstance(artifact_record, dict):
                            relative_path = artifact_record["path"]
                            artifact_metadata = artifact_record
                        else:
                            relative_path = artifact_record
                            artifact_metadata = {}
                        artifact = roots[galaxy] / relative_path
                        if not artifact.is_file():
                            raise FileNotFoundError(
                                f"Safe PSF checkpoint artifact is missing: {artifact}"
                            )
                        model, result, ensemble = read_psf_artifact(
                            artifact,
                            include_ensemble=True,
                        )
                        model.provenance.update(
                            artifact_metadata.get("provider_provenance", {})
                        )
                        model.quality.update(artifact_metadata.get("model_quality", {}))
                        ensemble.provenance.update(
                            artifact_metadata.get("ensemble_provenance", {})
                        )
                        ensemble.spatial_metrics.update(
                            artifact_metadata.get("ensemble_qc", {})
                        )
                        sample_metadata = artifact_metadata.get("sample_metadata", [])
                        sample_provenance = artifact_metadata.get("sample_provenance", [])
                        sample_quality = artifact_metadata.get("sample_quality", [])
                        for index, sample in enumerate(ensemble.samples):
                            if index < len(sample_metadata):
                                sample.metadata.update(sample_metadata[index])
                            if index < len(sample_provenance):
                                sample.model.provenance.update(sample_provenance[index])
                            if index < len(sample_quality):
                                sample.model.quality.update(sample_quality[index])
                        if result is not None:
                            result.metrics.update(artifact_metadata.get("match_qc", {}))
                            result.target_psf.provenance.update(
                                artifact_metadata.get("target_provenance", {})
                            )
                        image_set._safe_psf_models.setdefault(galaxy, {}).setdefault(
                            observatory, {}
                        )[band] = model
                        image_set._psf_ensembles.setdefault(galaxy, {}).setdefault(
                            observatory, {}
                        )[band] = ensemble
                        if result is not None:
                            image_set._psf_match.setdefault(galaxy, {}).setdefault(
                                observatory, {}
                            )[band] = result

            gaia = state.get("gaia_reference")
            if gaia is not None:
                cache = getattr(image_set, "_gaia_mask_reference_cache", None)
                if cache is None:
                    cache = {}
                    image_set._gaia_mask_reference_cache = cache
                cache[galaxy] = gaia

        return image_set

    def validate_step_plan(self, steps):
        current = [self._step_record(index, step) for index, step in enumerate(steps, start=1)]
        saved = next(iter(self.manifests.values())).get("step_plan", [])
        if current != saved:
            raise ValueError(
                "The current pipeline step plan does not match the checkpoint manifest"
            )

    @staticmethod
    def _step_record(index, step):
        return {
            "index": int(index),
            "id": str(step["id"]),
            "name": str(step["name"]),
            "scope": str(step.get("scope", "image")),
            "mutates": sorted(str(item) for item in step.get("mutates", ())),
        }

    @classmethod
    def step_plan(cls, steps):
        return [cls._step_record(index, step) for index, step in enumerate(steps, start=1)]

    def complete_step(self, step_index, step, products=None):
        step_id = str(step["id"])
        mutates = tuple(step.get("mutates", ()))
        checkpoints = {}
        if mutates:
            for galaxy in self.manifests:
                checkpoints[galaxy] = self._write_galaxy_checkpoint(
                    galaxy,
                    step_index,
                    step_id,
                    str(step["name"]),
                )

        product_records = self._product_records(products or [])
        for galaxy, manifest in self.manifests.items():
            completed = manifest.setdefault("completed_step_ids", [])
            if step_id not in completed:
                completed.append(step_id)
            if galaxy in checkpoints:
                manifest["latest_checkpoint"] = checkpoints[galaxy]
            manifest["state"] = self._state_for_galaxy(galaxy)
            manifest["products"].extend(
                record for record in product_records if record.get("galaxy") in {None, galaxy}
            )
            manifest["updated_at"] = _utc_now()
            _write_manifest_atomic(self.roots[galaxy] / "manifest.json", manifest)

    def _write_galaxy_checkpoint(self, galaxy, step_index, step_id, step_name):
        checkpoint_root = self.roots[galaxy] / "checkpoints"
        directory_name = f"{step_index:02d}-{_safe_component(step_id)}"
        final_dir = checkpoint_root / directory_name
        if final_dir.exists():
            raise FileExistsError(f"Checkpoint directory already exists: {final_dir}")
        temporary = checkpoint_root / f".{directory_name}.{uuid4().hex}.tmp"
        temporary.mkdir()
        files = []
        try:
            for observatory, bands in self.image_set.data[galaxy].items():
                for band, image in bands.items():
                    filename = "-".join(
                        _safe_component(item) for item in (galaxy, observatory, band)
                    ) + ".fits"
                    science_path = temporary / filename
                    error_path = temporary / f"{Path(filename).stem}_err.fits"
                    header = self._checkpoint_header(
                        galaxy,
                        observatory,
                        band,
                        step_index,
                        step_id,
                        step_name,
                    )
                    fits.PrimaryHDU(
                        np.asarray(image, dtype=np.float32),
                        header=header,
                    ).writeto(science_path, overwrite=False)
                    error = self.image_set.error.get(galaxy, {}).get(observatory, {}).get(band)
                    written_error = None
                    if error is not None:
                        fits.PrimaryHDU(
                            np.asarray(error, dtype=np.float32),
                            header=header,
                        ).writeto(error_path, overwrite=False)
                        written_error = error_path.name
                    files.append({
                        "observatory": str(observatory),
                        "band": str(band),
                        "science": f"checkpoints/{directory_name}/{science_path.name}",
                        "error": (
                            f"checkpoints/{directory_name}/{written_error}"
                            if written_error is not None else None
                        ),
                        "product_metadata": (
                            self.image_set.product_metadata[galaxy][observatory][band].to_dict()
                            if band in self.image_set.product_metadata.get(galaxy, {}).get(observatory, {})
                            else None
                        ),
                    })
            temporary.replace(final_dir)
        except Exception:
            # Leave a clearly marked temporary directory for diagnosis. It is
            # never referenced by the manifest and therefore never resumed.
            raise
        return {
            "step_id": step_id,
            "step_index": int(step_index),
            "directory": f"checkpoints/{directory_name}",
            "files": files,
        }

    def _checkpoint_header(self, galaxy, observatory, band, step_index, step_id, step_name):
        header = self.image_set.header[galaxy][observatory][band].copy()
        run_id = next(iter(self.manifests.values()))["run_id"]
        header["S7RUNID"] = (run_id, "Spec7DT checkpoint run")
        header["S7STEP"] = (step_id, "Last completed Spec7DT step")
        header["S7STPN"] = (int(step_index), "Spec7DT step number")
        header["S7VERS"] = (_package_version(), "Spec7DT version")
        source = self.image_set.get_error_source(galaxy, observatory, band)
        header["ERRSRC"] = (str(source), "Spec7DT error-map source")
        psf = self.image_set.psf.get(galaxy, {}).get(observatory, {}).get(band)
        try:
            psf_value = float(psf)
        except (TypeError, ValueError):
            psf_value = np.nan
        if np.isfinite(psf_value) and psf_value > 0:
            header["PSFIN"] = (psf_value, "Measured input PSF FWHM [arcsec]")
            if (
                header.get("PSFMODE") != "SAFE"
                and (
                    "psf_convolve" in next(iter(self.manifests.values()))["completed_step_ids"]
                    or step_id == "psf_convolve"
                )
            ):
                valid = []
                for value in (
                    item
                    for bands in self.image_set.psf.get(galaxy, {}).values()
                    if isinstance(bands, dict)
                    for item in bands.values()
                ):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value) and value > 0:
                        valid.append(value)
                if valid:
                    header["PSFOUT"] = (max(valid), "Matched PSF FWHM [arcsec]")
        header.add_history(f"Spec7DT completed step {step_index}: {step_name}")
        return header

    def _write_safe_psf_artifacts(self, galaxy):
        models = getattr(self.image_set, "_safe_psf_models", {}).get(galaxy, {})
        if not models:
            return {}
        from ..reduction.PSF_safe import write_psf_artifact

        matches = getattr(self.image_set, "_psf_match", {}).get(galaxy, {})
        records = {}
        root = self.roots[galaxy]
        for observatory, bands in models.items():
            for band, model in bands.items():
                filename = (
                    f"{_safe_component(observatory)}--{_safe_component(band)}.fits"
                )
                relative = Path("artifacts") / "psf" / filename
                result = matches.get(observatory, {}).get(band)
                ensemble = (
                    getattr(self.image_set, "_psf_ensembles", {})
                    .get(galaxy, {})
                    .get(observatory, {})
                    .get(band)
                )
                write_psf_artifact(root / relative, model, result, ensemble=ensemble)
                samples = [] if ensemble is None else ensemble.samples
                records.setdefault(str(observatory), {})[str(band)] = {
                    "path": str(relative),
                    "provider_provenance": _copy_tree(model.provenance),
                    "model_quality": _copy_tree(model.quality),
                    "ensemble_provenance": _copy_tree(
                        {} if ensemble is None else ensemble.provenance
                    ),
                    "ensemble_qc": _copy_tree(
                        {} if ensemble is None else ensemble.spatial_metrics
                    ),
                    "sample_metadata": [_copy_tree(sample.metadata) for sample in samples],
                    "sample_provenance": [
                        _copy_tree(sample.model.provenance) for sample in samples
                    ],
                    "sample_quality": [
                        _copy_tree(sample.model.quality) for sample in samples
                    ],
                    "match_qc": _copy_tree({} if result is None else result.metrics),
                    "target_provenance": _copy_tree(
                        {} if result is None else result.target_psf.provenance
                    ),
                }
        return records

    def _state_for_galaxy(self, galaxy):
        return {
            "psf": _copy_tree(self.image_set.psf.get(galaxy, {})),
            "cutout_shape": _copy_tree(self.image_set.cutout_shape.get(galaxy, {})),
            "error_source": _copy_tree(self.image_set.error_source.get(galaxy, {})),
            "gaia_reference": _checkpoint_gaia_state(self.image_set, galaxy),
            "safe_psf_failures": {
                key: value
                for key, value in getattr(self.image_set, "_safe_psf_failures", {}).items()
                if str(key).startswith(f"{galaxy}/")
            },
            "safe_psf_artifacts": self._write_safe_psf_artifacts(galaxy),
        }

    @staticmethod
    def _product_records(products):
        records = []
        seen_singletons = set()
        for item in products:
            product = item.get("product")
            if product is None:
                continue
            metadata = getattr(product, "metadata", {})
            kind = getattr(product, "kind", type(product).__name__)
            data = getattr(product, "data", product)
            if kind == "gaia_reference" and isinstance(data, dict):
                singleton_key = (item.get("galaxy"), kind)
                if singleton_key in seen_singletons:
                    continue
                seen_singletons.add(singleton_key)
                excluded = {"reference_image", "reference_header", "detections"}
                data = {
                    key: _copy_tree(value)
                    for key, value in data.items()
                    if key not in excluded
                }
            records.append({
                "galaxy": item.get("galaxy"),
                "observatory": item.get("observatory"),
                "band": item.get("band"),
                "kind": kind,
                "data": data,
                "metadata": metadata,
            })
        return records

    def _publish_output_dirs(self):
        setattr(self.image_set, "_pipeline_output_dirs", dict(self.roots))
