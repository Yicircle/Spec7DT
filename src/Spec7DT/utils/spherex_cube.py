from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..handlers.filter_handler import Filters
from ..handlers.spherex_response import (
    SPHEREX_CALIBRATION_VERSION,
    band_name_for_channel,
    register_spherex_filter,
)
from ..psf.models import ImageProductMetadata, make_product_id
from ..psf.products import sampling_from_header, wcs_metadata_from_header
from .reporting import emit_alert, emit_detail


PSF_COMMON_MODE = "CUBE-COMMON"
PSF_PLANE_MODE = "CUBE-PLANE"
PSF_MISMATCH_POLICIES = ("keep", "broaden", "drop")
# Noise-correlation block sizes written as NCORRF<n> band-header keys.
NOISE_CORRELATION_BLOCK_SIZES = tuple(range(2, 8))
_HOMOGENIZED_REQUIRED_HDUS = (
    "PSF_COVERAGE",
    "FLAGS",
    "PLANE_QUALITY",
    "PSF_INPUTS",
    "NOISE_CORR",
)
_PLANE_QUALITY_COLUMNS = {
    "PLANE",
    "PLANEQ",
    "QFLAGS",
    "ERRSCALE",
    "PSFQ",
    "PSF_R80_EXCESS",
    "SYS_FRAC",
}


_STRUCTURAL_KEYS = {
    "SIMPLE",
    "BITPIX",
    "EXTEND",
    "XTENSION",
    "PCOUNT",
    "GCOUNT",
    "CHECKSUM",
    "DATASUM",
    "EXTNAME",
}
_WCS_KEY = re.compile(
    r"^(?:WCSAXES|CRPIX\d+|CDELT\d+|CUNIT\d+|CTYPE\d+|CRVAL\d+|"
    r"CROTA\d+|CD\d+_\d+|PC\d+_\d+|PV\d+_\d+|PS\d+_\d+|"
    r"RADESYS|RADECSYS|EQUINOX|LONPOLE|LATPOLE|WCSNAME|MJDREF)[A-Z]?$"
)


def _is_mjy_per_pixel(value) -> bool:
    normalized = str(value).strip().lower().replace(" ", "")
    return normalized in {"mjy/pixel", "mjy/pix", "mjypixel-1", "mjypix-1"}


def _band_from_wavelength(wavelength_um: float) -> str:
    if not np.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError(f"Invalid SPHEREx channel wavelength: {wavelength_um}")
    wavelength_nm = int(np.floor(float(wavelength_um) * 1000.0 + 0.5))
    return f"m{wavelength_nm}"


def _selected_indices(channels, channel_count: int) -> list[int]:
    if channels is None:
        return list(range(channel_count))
    if isinstance(channels, slice):
        return list(range(channel_count))[channels]
    if isinstance(channels, (int, np.integer)) and not isinstance(channels, (bool, np.bool_)):
        channels = [int(channels)]

    try:
        indices = [int(value) for value in channels]
    except (TypeError, ValueError) as exc:
        raise TypeError("channels must be None, a slice, or zero-based integer indices") from exc
    if len(indices) != len(set(indices)):
        raise ValueError("channels contains duplicate indices")
    if any(index < 0 or index >= channel_count for index in indices):
        raise IndexError(
            f"SPHEREx channel indices must be within [0, {channel_count - 1}]"
        )
    return indices


def _slice_header(primary_header, image_header, hdul, shape, channel) -> fits.Header:
    from .file_handler import GalaxyImageSet

    merged = GalaxyImageSet._merge_primary_and_image_headers(
        primary_header,
        image_header,
    )
    cleaned = merged.copy()
    for key in list(cleaned.keys()):
        normalized = str(key).strip().upper()
        if (
            normalized in _STRUCTURAL_KEYS
            or re.fullmatch(r"NAXIS\d*", normalized)
            or _WCS_KEY.fullmatch(normalized)
        ):
            while key in cleaned:
                del cleaned[key]

    try:
        celestial = WCS(image_header, fobj=hdul, relax=True).celestial
        celestial_header = celestial.to_header(relax=True)
    except Exception as exc:
        raise ValueError(f"SPHEREx cube celestial WCS could not be extracted: {exc}") from exc
    if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
        raise ValueError("SPHEREx cube must contain exactly two celestial WCS axes")

    cleaned.update(celestial_header)
    cleaned["NAXIS"] = 2
    cleaned["NAXIS1"] = int(shape[1])
    cleaned["NAXIS2"] = int(shape[0])
    cleaned["BUNIT"] = ("mJy", "Flux density assigned to one output pixel")
    cleaned["ORIGUNIT"] = ("mJy/pixel", "Unit in source cube extension")
    cleaned["FILTER"] = (channel["band"], "Spec7DT SPHEREx channel label")
    cleaned["DETECTOR"] = (channel["detector"], "SPHEREx detector band")
    cleaned["SUBCHAN"] = (channel["subchannel"], "17-channel subchannel")
    cleaned["WAVELEN"] = (channel["wavelength"], "Nominal wavelength [um]")
    cleaned["WL_MIN"] = (channel["wl_min"], "Nominal lower wavelength [um]")
    cleaned["WL_MAX"] = (channel["wl_max"], "Nominal upper wavelength [um]")
    cleaned["BANDWID"] = (channel["bandwidth"], "Nominal bandwidth [um]")
    cleaned["SPEC_R"] = (channel["resolution"], "Nominal resolving power")
    cleaned["SPHPLN"] = (channel["index"] + 1, "One-based source cube plane")
    cleaned["SRCEXT"] = ("IMAGE_MJYPIX", "Source cube science extension")
    cleaned.add_history(
        "Spec7DT imported this 2-D band from a SPHEREx spectro-photometric mosaic."
    )
    return cleaned


def _header_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().upper() in {"T", "TRUE", "1", "Y", "YES"}


def psf_homogenization_mode(primary_header) -> str | None:
    """Return ``'CUBE-COMMON'`` for a supported homogenized cube, else ``None``.

    ``CUBE-PLANE`` products have plane-dependent resolution and cannot be used
    for pixel SEDs, so they are rejected.  A cube that claims ``PSFMATCH`` with
    any other mode is also rejected rather than treated as a native mosaic,
    because the native path would synthesize a pre-homogenization PSF for it.
    """

    mode = str(primary_header.get("PSFMODE", "") or "").strip().upper()
    if mode == PSF_PLANE_MODE:
        raise ValueError(
            "SPHEREx PSFMODE='CUBE-PLANE' cubes have plane-dependent resolution and "
            "cannot be used for pixel SEDs; use the CUBE-COMMON (*-PSFCOMMON) product"
        )
    if not _header_bool(primary_header.get("PSFMATCH")):
        return None
    if mode == PSF_COMMON_MODE:
        return PSF_COMMON_MODE
    raise ValueError(
        f"SPHEREx cube has PSFMATCH=T with unsupported PSFMODE={mode or None!r}; "
        f"only {PSF_COMMON_MODE!r} is supported"
    )


def spherex_prf(sigma_px: float, oversample: int = 5) -> np.ndarray:
    """Gaussian(sigma_px) convolved with a one-native-pixel box.

    The stamp is sampled at sub-pixel centres, has odd size, is centred on a
    native pixel centre, and sums to one.
    """

    sigma_px = float(sigma_px)
    oversample = int(oversample)
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError(f"PRF sigma must be finite and positive; got {sigma_px}")
    if oversample < 1 or oversample % 2 == 0:
        raise ValueError("PRF oversampling must be a positive odd integer")
    from scipy.special import erf

    half = int(np.ceil(8.0 * np.hypot(sigma_px, np.sqrt(1.0 / 12.0))))
    size = (2 * half + 1) * oversample
    x = (np.arange(size) - (size - 1) / 2.0) / oversample
    scale = np.sqrt(2.0) * sigma_px
    profile = 0.5 * (erf((x + 0.5) / scale) - erf((x - 0.5) / scale))
    prf = np.outer(profile, profile)
    return prf / prf.sum()


def block_correlation_factor(noise_corr: np.ndarray, size: int) -> float:
    """Var(sum of a size x size block) / (size^2 sigma^2); 1 means uncorrelated."""

    noise_corr = np.asarray(noise_corr, dtype=float)
    if not np.isfinite(noise_corr).any():
        return 1.0
    center = noise_corr.shape[0] // 2
    total = 0.0
    for dy in range(-(size - 1), size):
        for dx in range(-(size - 1), size):
            if (
                abs(dy) <= center
                and abs(dx) <= center
                and np.isfinite(noise_corr[center + dy, center + dx])
            ):
                total += (
                    (size - abs(dx))
                    * (size - abs(dy))
                    * noise_corr[center + dy, center + dx]
                )
    return max(total / size ** 2, 1.0)


def _finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _set_header(header, key, value, comment):
    """Set a card unless its value cannot be represented in a FITS header."""

    if isinstance(value, (float, np.floating)):
        value = _finite_or_none(value)
    if value is None:
        if key in header:
            del header[key]
        return
    if isinstance(value, np.generic):
        value = value.item()
    header[key] = (value, comment)


def _plane_quality_records(hdul, plane_count):
    table = hdul["PLANE_QUALITY"].data
    names = {str(name).upper(): str(name) for name in (table.columns.names or ())}
    missing = sorted(_PLANE_QUALITY_COLUMNS - set(names))
    if missing:
        raise ValueError("SPHEREx PLANE_QUALITY is missing columns: " + ", ".join(missing))
    records = {}
    for row in table:
        plane = int(row[names["PLANE"]])
        records[plane] = {
            "planeq": str(row[names["PLANEQ"]]).strip().upper(),
            "qflags": int(row[names["QFLAGS"]]),
            "errscale": _finite_or_none(row[names["ERRSCALE"]]),
            "psfq": str(row[names["PSFQ"]]).strip().upper(),
            "r80_excess": _finite_or_none(row[names["PSF_R80_EXCESS"]]),
            "sys_frac": _finite_or_none(row[names["SYS_FRAC"]]),
        }
    missing_planes = sorted(set(range(plane_count)) - set(records))
    if missing_planes:
        raise ValueError(
            f"SPHEREx PLANE_QUALITY has no row for zero-based planes {missing_planes}"
        )
    return records


def _target_sigma_pixel(hdul):
    table = hdul["PSF_INPUTS"].data
    names = {str(name).upper(): str(name) for name in (table.columns.names or ())}
    if "TARGET_SIGMA" not in names:
        raise ValueError("SPHEREx PSF_INPUTS has no TARGET_SIGMA column")
    if "MODE" in names:
        modes = {str(value).strip().upper() for value in table[names["MODE"]]}
        if modes != {PSF_COMMON_MODE}:
            raise ValueError(f"SPHEREx PSF_INPUTS modes {sorted(modes)} are not {PSF_COMMON_MODE}")
    sigma = np.asarray(table[names["TARGET_SIGMA"]], dtype=float)
    sigma = sigma[np.isfinite(sigma)]
    if sigma.size == 0 or np.any(sigma <= 0):
        raise ValueError("SPHEREx PSF_INPUTS TARGET_SIGMA must be finite and positive")
    if np.ptp(sigma) > 1e-6 * float(np.max(sigma)):
        raise ValueError(
            "SPHEREx CUBE-COMMON cube has more than one TARGET_SIGMA "
            f"({float(np.min(sigma)):.6g}-{float(np.max(sigma)):.6g} px)"
        )
    return float(np.median(sigma))


def _reject_mask(hdul, filepath):
    value = hdul["FLAGS"].header.get("REJMASK")
    if value is None:
        emit_alert(
            f"SPHEREx FLAGS HDU in {filepath} has no REJMASK; flag-based pixel "
            "rejection was skipped",
            context="SPHEREx import",
            dedupe_key=("spherex-rejmask", str(filepath)),
        )
        return None
    return np.uint64(int(value))


def _invalid_pixels(science_plane, coverage_plane, flags_plane, coverage_threshold, reject_mask):
    invalid = ~np.isfinite(science_plane)
    coverage = np.asarray(coverage_plane, dtype=float)
    invalid |= ~(coverage >= coverage_threshold)
    if reject_mask is not None:
        flags = np.asarray(flags_plane).astype(np.uint64, copy=False)
        invalid |= (flags & reject_mask) != 0
    return invalid


def _effective_wavelength_summary(hdul, index, valid):
    if "WAVELENGTH_EFF" not in hdul or not np.any(valid):
        return None, None
    values = np.asarray(hdul["WAVELENGTH_EFF"].data[index], dtype=float)[valid]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    p1, p50, p99 = np.percentile(values, [1.0, 50.0, 99.0])
    return float(p50), float(p99 - p1)


def _homogenized_header(header, primary_header, quality, target_sigma, noise_corr, wavelength):
    for key in ("PSFMATCH", "PSFMODE"):
        while key in header:
            del header[key]
    header["SPXPSFH"] = (True, "SPHEREx cube is PSF homogenized")
    header["SPXPSFMD"] = (
        str(primary_header.get("PSFMODE", PSF_COMMON_MODE)).strip(),
        "Source SPHEREx PSFMODE",
    )
    _set_header(header, "SPXTSIG", target_sigma, "Target Gaussian sigma [native px]")
    _set_header(header, "SPXPLQ", quality["planeq"], "SPHEREx plane quality PLANEQ")
    _set_header(header, "SPXQFLG", quality["qflags"], "SPHEREx plane QFLAGS")
    _set_header(header, "SPXSYS", quality["sys_frac"], "Fractional systematic error")
    _set_header(header, "SPXERRSC", quality["errscale"], "Empirical error scale; not applied")
    _set_header(header, "SPXPSFQ", quality["psfq"], "SPHEREx plane PSF quality")
    _set_header(header, "SPXR80X", quality["r80_excess"], "Fractional r80 excess over target")
    for size in NOISE_CORRELATION_BLOCK_SIZES:
        _set_header(
            header,
            f"NCORRF{size}",
            block_correlation_factor(noise_corr, size),
            f"Noise variance factor of a {size}x{size} block sum",
        )
    sampling = sampling_from_header(header)
    _set_header(
        header,
        "NCORRSC",
        None if sampling is None else float(sampling.equivalent_scale_arcsec),
        "Pixel scale [arcsec] valid for NCORRF*",
    )
    median, spread = wavelength
    _set_header(header, "SPXLAMEF", median, "Median effective wavelength [um]")
    _set_header(header, "SPXLAMSP", spread, "P1-P99 effective wavelength spread [um]")
    return header


class SpherexMosaicImporter:
    """Import an IRSA-style SPHEREx mosaic cube as ordinary 2-D bands."""

    science_extension = "IMAGE_MJYPIX"
    error_extension = "ERROR_MJYPIX"
    channel_extension = "SPECTRAL_CHANNELS"
    required_channel_columns = {
        "DETECTOR",
        "SUBCHAN",
        "WAVELENGTH",
        "WL_MIN",
        "WL_MAX",
        "R",
        "BANDWIDTH",
    }

    def __init__(self, *, response_root=None, strict_version: bool = True):
        self.response_root = response_root
        self.strict_version = bool(strict_version)

    def load(
        self,
        filepath,
        *,
        galaxy_name: str | None = None,
        channels=None,
        drop_flagged_planes: bool = True,
        psf_mismatch_policy: str = "keep",
    ):
        """Load a SPHEREx mosaic cube as 2-D bands.

        Native mosaics are imported unchanged.  PSF-homogenized
        (``PSFMODE='CUBE-COMMON'``) cubes are screened and masked:

        * ``drop_flagged_planes`` drops ``PLANEQ='HE1083'`` planes and planes
          without a finite pixel unless ``channels`` selects planes explicitly,
          in which case the selection is kept and a warning is emitted.
        * ``psf_mismatch_policy`` controls ``PSFQ='MISMATCH'`` planes:
          ``'keep'`` (warn), ``'broaden'`` (the PSF provider widens sigma to
          the measured r80 excess), or ``'drop'``.

        Dropped planes and reasons are stored on the returned image set as
        ``_spherex_dropped_planes`` (``{galaxy: [{"plane", "band", "reasons"}]}``).
        """

        from .file_handler import GalaxyImageSet, Parsers

        if psf_mismatch_policy not in PSF_MISMATCH_POLICIES:
            raise ValueError(
                f"psf_mismatch_policy must be one of {PSF_MISMATCH_POLICIES}; "
                f"got {psf_mismatch_policy!r}"
            )
        filepath = Path(filepath).expanduser()
        if not filepath.is_file():
            raise FileNotFoundError(f"SPHEREx mosaic not found: {filepath}")

        with fits.open(filepath, memmap=True) as hdul:
            self._validate_extensions(hdul, filepath)
            science_hdu = hdul[self.science_extension]
            error_hdu = hdul[self.error_extension]
            channel_hdu = hdul[self.channel_extension]
            self._validate_version(hdul[0].header, channel_hdu.header, filepath)
            self._validate_units(science_hdu.header, error_hdu.header, filepath)
            homogenized = psf_homogenization_mode(hdul[0].header) == PSF_COMMON_MODE

            science = science_hdu.data
            error = error_hdu.data
            channel_table = channel_hdu.data
            if science is None or error is None or channel_table is None:
                raise ValueError(f"SPHEREx mosaic contains an empty required HDU: {filepath}")
            if science.ndim != 3 or error.ndim != 3:
                raise ValueError(
                    "SPHEREx IMAGE_MJYPIX and ERROR_MJYPIX must both be 3-D; "
                    f"got {science.shape} and {error.shape}"
                )
            if science.shape != error.shape:
                raise ValueError(
                    f"SPHEREx science/error cube shape mismatch: {science.shape} != {error.shape}"
                )
            if len(channel_table) != science.shape[0]:
                raise ValueError(
                    "SPHEREx SPECTRAL_CHANNELS row count does not match cube planes: "
                    f"{len(channel_table)} != {science.shape[0]}"
                )

            homogenized_state = None
            dropped = []
            if homogenized:
                homogenized_state = self._homogenized_state(hdul, filepath, science.shape)
                selected, dropped = self._screen_planes(
                    channels,
                    science,
                    homogenized_state["quality"],
                    drop_flagged_planes=drop_flagged_planes,
                    psf_mismatch_policy=psf_mismatch_policy,
                    filepath=filepath,
                )
            else:
                selected = _selected_indices(channels, science.shape[0])
            channel_specs = self._channel_specs(channel_table, selected)
            resolved_galaxy = galaxy_name or Parsers._galaxy_name_parser(filepath.stem)
            if not resolved_galaxy:
                raise ValueError(
                    "Could not infer a galaxy name from the cube filename; "
                    "provide galaxy_name explicitly."
                )
            if dropped:
                dropped_specs = self._channel_specs(
                    channel_table, [item["plane"] for item in dropped]
                )
                for item, spec in zip(dropped, dropped_specs):
                    item["band"] = spec["band"]

            # Resolve all calibration inputs before allocating and mutating the
            # returned image set, so malformed cubes fail atomically.
            for channel in channel_specs:
                register_spherex_filter(
                    Filters,
                    detector=channel["detector"],
                    subchannel=channel["subchannel"],
                    band_name=channel["band"],
                    response_root=self.response_root,
                )

            image_set = GalaxyImageSet(auto_ensure_filters=False)
            product_group_id = make_product_id(
                filepath,
                self.science_extension,
                None,
                "spherex-mosaic",
            )
            for channel in channel_specs:
                index = channel["index"]
                image = np.array(science[index], dtype=np.float32, copy=True)
                uncertainty = np.array(error[index], dtype=np.float32, copy=True)
                header = _slice_header(
                    hdul[0].header,
                    science_hdu.header,
                    hdul,
                    image.shape,
                    channel,
                )
                provenance = {
                    "adapter": "spherex-mosaic-cube",
                    "spectral_channel_hdu": self.channel_extension,
                    "subchannel": channel["subchannel"],
                    "source_plane": int(index),
                    "nhits_hdu": "NHITS" if "NHITS" in hdul else None,
                    "inputs_hdu": "INPUTS" if "INPUTS" in hdul else None,
                    "wl_min_um": channel["wl_min"],
                    "wl_max_um": channel["wl_max"],
                }
                if homogenized:
                    image, uncertainty, header = self._apply_homogenized_plane(
                        hdul,
                        index,
                        image,
                        uncertainty,
                        header,
                        homogenized_state,
                    )
                    quality = homogenized_state["quality"][index]
                    provenance.update({
                        "psf_homogenized": True,
                        "target_sigma_pixel": homogenized_state["target_sigma"],
                        "psf_grid_hdu": "PSF_GRID" if "PSF_GRID" in hdul else None,
                        "psf_tiles_hdu": "PSF_TILES" if "PSF_TILES" in hdul else None,
                        "psf_quality": {
                            "psfq": quality["psfq"],
                            "r80_excess": quality["r80_excess"],
                        },
                        "psf_mismatch_policy": psf_mismatch_policy,
                        "registration_check": "unresampled",
                    })
                image_set.add_array(
                    image,
                    header,
                    galaxy_name=resolved_galaxy,
                    observatory="SPHEREx",
                    band=channel["band"],
                    error=uncertainty,
                    source_path=filepath,
                    error_source="file",
                    ensure_filter=False,
                    product_metadata=ImageProductMetadata(
                        product_kind="coadd",
                        facility="SPHEREx",
                        instrument="SPHEREx mosaic",
                        release=str(hdul[0].header.get("SPCVER", "")) or None,
                        version=str(hdul[0].header.get("SPCVER", "")) or None,
                        detector=channel["detector"],
                        channel=channel["band"],
                        source_path=str(filepath),
                        hdu=self.science_extension,
                        plane=int(index),
                        product_group_id=product_group_id,
                        sampling=sampling_from_header(header),
                        output_wcs=wcs_metadata_from_header(header),
                        requires_output_effective_psf=True,
                        provenance=provenance,
                    ),
                )
        if homogenized:
            image_set._spherex_dropped_planes = {resolved_galaxy: dropped}
            if dropped:
                summary = ", ".join(
                    f"{item['plane']}({item['band']}: {'+'.join(item['reasons'])})"
                    for item in dropped
                )
                emit_alert(
                    f"Dropped {len(dropped)} SPHEREx planes from {filepath.name}: {summary}",
                    level="info",
                    context=f"{resolved_galaxy}/SPHEREx",
                    dedupe_key=("spherex-dropped-planes", str(filepath)),
                )
            emit_detail(
                f"SPHEREx PSF-homogenized cube {filepath.name}: imported "
                f"{len(channel_specs)} planes, dropped {len(dropped)}, "
                f"target sigma={homogenized_state['target_sigma']:.6g} px"
            )
        return image_set

    @classmethod
    def _homogenized_state(cls, hdul, filepath, cube_shape):
        missing = [name for name in _HOMOGENIZED_REQUIRED_HDUS if name not in hdul]
        if missing:
            raise ValueError(
                f"PSF-homogenized SPHEREx cube {filepath} is missing HDUs: {', '.join(missing)}"
            )
        for name in ("PSF_COVERAGE", "FLAGS"):
            shape = tuple(hdul[name].shape)
            if shape != tuple(cube_shape):
                raise ValueError(
                    f"SPHEREx {name} shape {shape} does not match the science cube {tuple(cube_shape)}"
                )
        noise_shape = tuple(hdul["NOISE_CORR"].shape)
        if len(noise_shape) != 3 or noise_shape[0] != cube_shape[0]:
            raise ValueError(f"SPHEREx NOISE_CORR shape {noise_shape} does not match the cube")
        threshold = _finite_or_none(hdul[0].header.get("PSFCOVTH", 0.999))
        if threshold is None:
            threshold = 0.999
        return {
            "quality": _plane_quality_records(hdul, int(cube_shape[0])),
            "target_sigma": _target_sigma_pixel(hdul),
            "coverage_threshold": float(threshold),
            "reject_mask": _reject_mask(hdul, filepath),
            # FLAGS is a BZERO-scaled uint32 cube, which cannot be memory-mapped.
            "flags": fits.getdata(filepath, "FLAGS", memmap=False),
        }

    @staticmethod
    def _screen_planes(
        channels,
        science,
        quality,
        *,
        drop_flagged_planes,
        psf_mismatch_policy,
        filepath,
    ):
        plane_count = int(science.shape[0])
        reasons = {}
        for plane in range(plane_count):
            plane_reasons = []
            if drop_flagged_planes:
                if quality[plane]["planeq"] == "HE1083":
                    plane_reasons.append("HE1083")
                if not np.isfinite(science[plane]).any():
                    plane_reasons.append("empty")
            if psf_mismatch_policy == "drop" and quality[plane]["psfq"] == "MISMATCH":
                plane_reasons.append("PSF_MISMATCH")
            if plane_reasons:
                reasons[plane] = plane_reasons

        if channels is not None:
            selected = _selected_indices(channels, plane_count)
            for plane in selected:
                if plane in reasons:
                    emit_alert(
                        f"explicitly selected SPHEREx plane {plane} would have been dropped "
                        f"({'+'.join(reasons[plane])}); it is kept",
                        context=f"SPHEREx import {Path(filepath).name}",
                        dedupe_key=("spherex-selected-flagged", str(filepath), plane),
                    )
            dropped = []
        else:
            selected = [plane for plane in range(plane_count) if plane not in reasons]
            dropped = [
                {"plane": plane, "reasons": list(reasons[plane])}
                for plane in sorted(reasons)
            ]
        if not selected:
            raise ValueError(f"No usable SPHEREx planes remain in {filepath}")

        if psf_mismatch_policy == "keep":
            mismatched = [plane for plane in selected if quality[plane]["psfq"] == "MISMATCH"]
            if mismatched:
                emit_alert(
                    f"{len(mismatched)} SPHEREx planes have PSFQ='MISMATCH' (planes "
                    f"{mismatched[0]}-{mismatched[-1]}); they keep the target PSF "
                    "(psf_mismatch_policy='keep')",
                    context=f"SPHEREx import {Path(filepath).name}",
                    dedupe_key=("spherex-psf-mismatch", str(filepath)),
                )
        return selected, dropped

    @staticmethod
    def _apply_homogenized_plane(hdul, index, image, uncertainty, header, state):
        invalid = _invalid_pixels(
            image,
            hdul["PSF_COVERAGE"].data[index],
            state["flags"][index],
            state["coverage_threshold"],
            state["reject_mask"],
        )
        image[invalid] = np.nan
        uncertainty[invalid] = np.nan
        header = _homogenized_header(
            header,
            hdul[0].header,
            state["quality"][index],
            state["target_sigma"],
            np.asarray(hdul["NOISE_CORR"].data[index], dtype=float),
            _effective_wavelength_summary(hdul, index, ~invalid),
        )
        return image, uncertainty, header

    @classmethod
    def _validate_extensions(cls, hdul, filepath):
        missing = [
            name
            for name in (
                cls.science_extension,
                cls.error_extension,
                cls.channel_extension,
            )
            if name not in hdul
        ]
        if missing:
            raise ValueError(
                f"SPHEREx mosaic {filepath} is missing required HDUs: {', '.join(missing)}"
            )
        columns = set(hdul[cls.channel_extension].columns.names or ())
        missing_columns = sorted(cls.required_channel_columns - columns)
        if missing_columns:
            raise ValueError(
                "SPHEREx SPECTRAL_CHANNELS is missing columns: "
                + ", ".join(missing_columns)
            )

    def _validate_version(self, primary_header, channel_header, filepath):
        version = primary_header.get("SPCVER") or channel_header.get("SPCVER")
        if version is None:
            if self.strict_version:
                raise ValueError(f"SPHEREx mosaic {filepath} has no SPCVER calibration version")
            return
        if self.strict_version and SPHEREX_CALIBRATION_VERSION not in str(version):
            raise ValueError(
                f"SPHEREx mosaic calibration {version!r} is incompatible with packaged "
                f"responses {SPHEREX_CALIBRATION_VERSION}"
            )

    @staticmethod
    def _validate_units(science_header, error_header, filepath):
        for extension, header in (
            ("IMAGE_MJYPIX", science_header),
            ("ERROR_MJYPIX", error_header),
        ):
            if not _is_mjy_per_pixel(header.get("BUNIT")):
                raise ValueError(
                    f"SPHEREx {extension} in {filepath} must use mJy/pixel; "
                    f"got {header.get('BUNIT')!r}"
                )

    @staticmethod
    def _channel_specs(channel_table, indices):
        specs = []
        channel_keys = set()
        band_names = set()
        for index in indices:
            row = channel_table[index]
            detector = int(row["DETECTOR"])
            subchannel = int(row["SUBCHAN"])
            band = _band_from_wavelength(float(row["WAVELENGTH"]))
            expected = band_name_for_channel(detector, subchannel)
            if band.lower() != expected.lower():
                raise ValueError(
                    "SPHEREx SPECTRAL_CHANNELS does not match the 202607 response index: "
                    f"plane={index + 1}, detector={detector}, subchannel={subchannel}, "
                    f"cube={band}, expected={expected}"
                )
            key = (detector, subchannel)
            if key in channel_keys:
                raise ValueError(f"Duplicate SPHEREx detector/subchannel in cube: {key}")
            if band.lower() in band_names:
                raise ValueError(f"Duplicate SPHEREx band label in cube: {band}")
            channel_keys.add(key)
            band_names.add(band.lower())
            specs.append({
                "index": int(index),
                "detector": detector,
                "subchannel": subchannel,
                "band": band,
                "wavelength": float(row["WAVELENGTH"]),
                "wl_min": float(row["WL_MIN"]),
                "wl_max": float(row["WL_MAX"]),
                "resolution": float(row["R"]),
                "bandwidth": float(row["BANDWIDTH"]),
            })
        return specs


__all__ = ["SpherexMosaicImporter"]
