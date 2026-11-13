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

    def load(self, filepath, *, galaxy_name: str | None = None, channels=None):
        from .file_handler import GalaxyImageSet, Parsers

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

            selected = _selected_indices(channels, science.shape[0])
            channel_specs = self._channel_specs(channel_table, selected)
            resolved_galaxy = galaxy_name or Parsers._galaxy_name_parser(filepath.stem)
            if not resolved_galaxy:
                raise ValueError(
                    "Could not infer a galaxy name from the cube filename; "
                    "provide galaxy_name explicitly."
                )

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
                        provenance={
                            "adapter": "spherex-mosaic-cube",
                            "spectral_channel_hdu": self.channel_extension,
                            "subchannel": channel["subchannel"],
                            "source_plane": int(index),
                            "nhits_hdu": "NHITS" if "NHITS" in hdul else None,
                            "inputs_hdu": "INPUTS" if "INPUTS" in hdul else None,
                            "wl_min_um": channel["wl_min"],
                            "wl_max_um": channel["wl_max"],
                        },
                    ),
                )
        return image_set

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
