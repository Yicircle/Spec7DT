"""Data model and extraction helpers for interactive pixel SED drilling.

This module deliberately has no notebook or widget dependency.  Frontends can
use :class:`SedDrillSession` with a live ``GalaxyImageSet`` or a catalogue
written with ``include_drill_metadata=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .utils.file_generator import galaxy_filter_columns


_PIXEL_ID_PATTERN = re.compile(r"^(?P<galaxy>.+)_(?P<pixel>\d+)$")
_DEFAULT_INVALID_VALUES = (99.0, -99.0)
_SELECTION_COLORS = (
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#e377c2",
    "#8c564b",
)


@dataclass(frozen=True)
class BandInfo:
    """Metadata for one observed filter in a drilling session."""

    observatory: str
    band: str
    flux_column: str | None = None
    error_column: str | None = None
    pivot_wavelength: float | None = None
    center_wavelength: float | None = None
    mean_wavelength: float | None = None
    fwhm: float | None = None
    unit: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.observatory, self.band)

    @property
    def wavelength_microns(self) -> float | None:
        if self.pivot_wavelength is None:
            return None
        return self.pivot_wavelength / 1.0e4

    @property
    def label(self) -> str:
        wavelength = self.wavelength_microns
        prefix = "unknown wavelength" if wavelength is None else f"{wavelength:.4g} μm"
        return f"{prefix} — {self.observatory}/{self.band}"


@dataclass(frozen=True)
class PixelSelection:
    """One pixel selected for display and comparison."""

    selection_id: int
    x: int
    y: int
    flat_index: int
    color: str
    ra_deg: float | None = None
    dec_deg: float | None = None


@dataclass(frozen=True)
class PixelSpectrum:
    """Raw pixel values and display-validity masks aligned with ``bands``."""

    selection: PixelSelection
    bands: tuple[BandInfo, ...]
    flux: np.ndarray
    error: np.ndarray
    valid_flux: np.ndarray
    valid_error: np.ndarray


class SedDrillSession:
    """A validated single-galaxy source for interactive pixel SED exploration."""

    max_selections = len(_SELECTION_COLORS)

    def __init__(
        self,
        *,
        galaxy: str,
        bands: Iterable[BandInfo],
        shape: tuple[int, int],
        images: Mapping[tuple[str, str], np.ndarray] | None = None,
        errors: Mapping[tuple[str, str], np.ndarray | None] | None = None,
        headers: Mapping[tuple[str, str], fits.Header] | None = None,
        reference_header: fits.Header | None = None,
        catalog_data: pd.DataFrame | None = None,
        invalid_values: Iterable[float] = _DEFAULT_INVALID_VALUES,
        source: str = "image_set",
    ):
        self.galaxy = str(galaxy)
        self.bands = tuple(bands)
        if not self.bands:
            raise ValueError("At least one filter is required for SED drilling.")

        self.shape = self._validate_shape(shape)
        self.source = str(source)
        self.invalid_values = tuple(float(value) for value in invalid_values)
        self._images = dict(images or {})
        self._errors = dict(errors or {})
        self._headers = dict(headers or {})
        self._catalog_data = catalog_data.copy(deep=True) if catalog_data is not None else None
        self._catalog_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._selections: list[PixelSelection] = []
        self._next_selection_id = 1
        self._active_band_index = 0
        self._reference_header = reference_header.copy() if reference_header is not None else None
        self._reference_wcs = self._optional_wcs(self._reference_header)

    @classmethod
    def from_image_set(
        cls,
        image_set,
        galaxy: str,
        *,
        strict_wcs: bool = True,
        invalid_values: Iterable[float] = _DEFAULT_INVALID_VALUES,
    ) -> "SedDrillSession":
        """Create a session from one processed galaxy in ``GalaxyImageSet``.

        The input images are not copied.  They must already occupy a common
        final grid; this constructor only validates the grid and never resamples
        data.
        """
        galaxy = str(galaxy)
        data_tree = getattr(image_set, "data", {})
        if galaxy not in data_tree:
            available = ", ".join(sorted(map(str, data_tree))) or "none"
            raise ValueError(f"Galaxy '{galaxy}' is not present. Available galaxies: {available}.")

        catalog_frame = getattr(image_set, "last_catalog", None)
        catalog_type = getattr(catalog_frame, "catalog_type", None)
        mapped_columns: dict[tuple[str, str], tuple[str, str]] = {}
        if catalog_type:
            for observatory, band, flux_column, error_column in galaxy_filter_columns(
                image_set, galaxy, catalog_type
            ):
                mapped_columns[(str(observatory), str(band))] = (
                    str(flux_column),
                    str(error_column),
                )

        image_map: dict[tuple[str, str], np.ndarray] = {}
        error_map: dict[tuple[str, str], np.ndarray | None] = {}
        header_map: dict[tuple[str, str], fits.Header] = {}
        band_info: list[BandInfo] = []
        expected_shape: tuple[int, int] | None = None
        reference_key: tuple[str, str] | None = None
        reference_header: fits.Header | None = None
        common_units: set[str] = set()

        for observatory, observatory_bands in data_tree[galaxy].items():
            for band, values in observatory_bands.items():
                key = (str(observatory), str(band))
                image = np.asarray(values)
                if image.ndim != 2 or image.size == 0:
                    raise ValueError(f"{galaxy}/{observatory}/{band} must be a non-empty 2-D image.")
                if expected_shape is None:
                    expected_shape = tuple(int(value) for value in image.shape)
                    reference_key = key
                elif image.shape != expected_shape:
                    raise ValueError(
                        f"Image shape mismatch for {galaxy}/{observatory}/{band}: "
                        f"expected {expected_shape}, got {image.shape}."
                    )

                try:
                    header = image_set.header[galaxy][observatory][band]
                    header = header.copy()
                except (AttributeError, KeyError) as exc:
                    raise ValueError(f"Header is missing for {galaxy}/{observatory}/{band}.") from exc
                if reference_header is None:
                    reference_header = header.copy()
                header_map[key] = header
                image_map[key] = image

                error = getattr(image_set, "error", {}).get(galaxy, {}).get(observatory, {}).get(band)
                if error is None:
                    error_map[key] = None
                else:
                    error_array = np.asarray(error)
                    if error_array.shape != image.shape:
                        raise ValueError(
                            f"Error-map shape mismatch for {galaxy}/{observatory}/{band}: "
                            f"science={image.shape}, error={error_array.shape}."
                        )
                    error_map[key] = error_array

                unit = str(header.get("BUNIT", "")).strip() or None
                if unit is not None:
                    common_units.add(unit)
                flux_column, error_column = mapped_columns.get(
                    key, (f"{observatory}.{band}", f"{observatory}.{band}_err")
                )
                band_info.append(
                    cls._band_info_from_filter(
                        image_set,
                        observatory=str(observatory),
                        band=str(band),
                        flux_column=flux_column,
                        error_column=error_column,
                        unit=unit,
                    )
                )

        if expected_shape is None or reference_key is None or reference_header is None:
            raise ValueError(f"Galaxy '{galaxy}' has no images for SED drilling.")
        if len(common_units) > 1:
            raise ValueError(
                "SED drilling requires compatible image BUNIT values; found "
                + ", ".join(sorted(common_units))
                + "."
            )
        if not common_units:
            warnings.warn(
                "Image BUNIT values are absent; SED drilling cannot validate flux units.",
                RuntimeWarning,
                stacklevel=2,
            )

        cls._validate_wcs_alignment(
            reference_header,
            header_map,
            expected_shape,
            strict_wcs=strict_wcs,
            reference_key=reference_key,
        )
        cls._warn_if_psf_status_unknown(image_set, galaxy)
        ordered = cls._sort_bands(band_info)
        return cls(
            galaxy=galaxy,
            bands=ordered,
            shape=expected_shape,
            images=image_map,
            errors=error_map,
            headers=header_map,
            reference_header=reference_header,
            invalid_values=invalid_values,
            source="image_set",
        )

    @classmethod
    def from_catalog(
        cls,
        catalog_path: str | Path,
        galaxy: str | None = None,
        *,
        metadata_path: str | Path | None = None,
    ) -> "SedDrillSession":
        """Create a lazy, catalogue-backed session from a drilling catalogue.

        The catalogue must have been emitted with
        ``write_catalog(include_drill_metadata=True)``.  For FITS the metadata
        lives in extensions; ASCII formats use an adjacent JSON sidecar.
        """
        catalog_path = Path(catalog_path)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

        if catalog_path.suffix.lower() in {".fits", ".fit", ".fts"}:
            metadata, catalog_data = cls._read_fits_catalog(catalog_path, galaxy)
        else:
            metadata, catalog_data = cls._read_ascii_catalog(
                catalog_path, galaxy, metadata_path
            )

        selected_galaxy = str(metadata["galaxy"])
        shape_values = metadata.get("shape")
        if not isinstance(shape_values, (list, tuple)) or len(shape_values) != 2:
            raise ValueError("Drilling metadata has no valid 2-D image shape.")
        header_text = metadata.get("reference_header", "")
        header = cls._header_from_string(header_text)
        bands = [cls._band_info_from_metadata(item, metadata.get("flux_unit")) for item in metadata.get("filters", [])]
        if not bands:
            raise ValueError("Drilling metadata contains no filter definitions.")
        return cls(
            galaxy=selected_galaxy,
            bands=cls._sort_bands(bands),
            shape=(int(shape_values[0]), int(shape_values[1])),
            reference_header=header,
            catalog_data=catalog_data,
            invalid_values=metadata.get("invalid_values", _DEFAULT_INVALID_VALUES),
            source="catalog",
        )

    @property
    def selections(self) -> tuple[PixelSelection, ...]:
        """Current selections in reference-pixel order."""
        return tuple(self._selections)

    @property
    def active_band_index(self) -> int:
        return self._active_band_index

    @property
    def active_band(self) -> BandInfo:
        return self.bands[self._active_band_index]

    @property
    def reference_selection(self) -> PixelSelection | None:
        return self._selections[0] if self._selections else None

    def set_filter(self, index: int) -> BandInfo:
        """Select a filter by its wavelength-sorted index."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Filter index must be an integer.")
        if not 0 <= int(index) < len(self.bands):
            raise IndexError(f"Filter index must be between 0 and {len(self.bands) - 1}.")
        self._active_band_index = int(index)
        return self.active_band

    def image_for_band(self, band: int | BandInfo | None = None) -> np.ndarray:
        """Return the science image for a band, lazily rebuilding catalogue maps."""
        info = self._resolve_band(band)
        key = info.key
        if self.source == "catalog":
            return self._catalog_maps_for_band(info)[0]
        return self._images[key]

    def error_for_band(self, band: int | BandInfo | None = None) -> np.ndarray | None:
        """Return the error image for a band, or ``None`` when unavailable."""
        info = self._resolve_band(band)
        if self.source == "catalog":
            return self._catalog_maps_for_band(info)[1]
        return self._errors.get(info.key)

    def select_pixel(self, x: float, y: float, *, mode: str = "replace") -> PixelSelection:
        """Select the nearest pixel, replacing or adding to current selections."""
        if mode not in {"replace", "add"}:
            raise ValueError("mode must be 'replace' or 'add'.")
        x_index, y_index = self._coerce_pixel(x, y)
        if mode == "replace":
            self.clear_selections()
        else:
            for selection in self._selections:
                if selection.x == x_index and selection.y == y_index:
                    return selection
        if len(self._selections) >= self.max_selections:
            raise ValueError(
                f"At most {self.max_selections} pixels can be compared in one session."
            )

        selection = PixelSelection(
            selection_id=self._next_selection_id,
            x=x_index,
            y=y_index,
            flat_index=y_index * self.shape[1] + x_index,
            color=_SELECTION_COLORS[len(self._selections)],
            **self._sky_position(x_index, y_index),
        )
        self._next_selection_id += 1
        self._selections.append(selection)
        return selection

    def remove_selection(self, selection: PixelSelection | int) -> PixelSelection:
        """Remove a selected pixel by object or selection ID."""
        selection_id = selection.selection_id if isinstance(selection, PixelSelection) else int(selection)
        for index, candidate in enumerate(self._selections):
            if candidate.selection_id == selection_id:
                return self._selections.pop(index)
        raise KeyError(f"Selection {selection_id} is not present.")

    def clear_selections(self) -> None:
        """Clear all selected pixels while keeping the current filter."""
        self._selections.clear()

    def spectrum_for_selection(self, selection: PixelSelection | int) -> PixelSpectrum:
        """Extract raw flux/error samples and validity flags for one pixel."""
        resolved = self._resolve_selection(selection)
        flux = np.empty(len(self.bands), dtype=float)
        error = np.empty(len(self.bands), dtype=float)
        for index, info in enumerate(self.bands):
            image = self.image_for_band(info)
            uncertainty = self.error_for_band(info)
            flux[index] = float(image[resolved.y, resolved.x])
            error[index] = (
                np.nan if uncertainty is None else float(uncertainty[resolved.y, resolved.x])
            )
        invalid = np.asarray(self.invalid_values, dtype=float)
        valid_flux = np.isfinite(flux) & ~np.isin(flux, invalid)
        valid_error = np.isfinite(error) & (error >= 0) & ~np.isin(error, invalid)
        return PixelSpectrum(resolved, self.bands, flux, error, valid_flux, valid_error)

    def comparison_spectrum(
        self,
        selection: PixelSelection | int,
        *,
        mode: str = "absolute",
    ) -> PixelSpectrum:
        """Return an absolute or first-selection-relative spectrum."""
        spectrum = self.spectrum_for_selection(selection)
        if mode == "absolute":
            return spectrum
        if mode != "ratio":
            raise ValueError("mode must be 'absolute' or 'ratio'.")
        reference = self.reference_selection
        if reference is None:
            raise ValueError("Select a reference pixel before requesting a ratio.")
        if spectrum.selection.selection_id == reference.selection_id:
            flux = np.where(spectrum.valid_flux, 1.0, np.nan)
            error = np.where(spectrum.valid_flux, 0.0, np.nan)
            return PixelSpectrum(
                spectrum.selection,
                spectrum.bands,
                flux,
                error,
                np.isfinite(flux),
                np.isfinite(error),
            )

        reference_spectrum = self.spectrum_for_selection(reference)
        denominator = reference_spectrum.flux
        valid_flux = (
            spectrum.valid_flux
            & reference_spectrum.valid_flux
            & (denominator != 0)
        )
        flux = np.full(len(self.bands), np.nan, dtype=float)
        np.divide(spectrum.flux, denominator, out=flux, where=valid_flux)

        valid_error = valid_flux & spectrum.valid_error & reference_spectrum.valid_error
        error = np.full(len(self.bands), np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            propagated = np.sqrt(
                (spectrum.error / denominator) ** 2
                + (spectrum.flux * reference_spectrum.error / denominator ** 2) ** 2
            )
        error[valid_error] = propagated[valid_error]
        valid_error &= np.isfinite(error)
        error[~valid_error] = np.nan
        return PixelSpectrum(spectrum.selection, spectrum.bands, flux, error, valid_flux, valid_error)

    def comparison_spectra(self, *, mode: str = "absolute") -> tuple[PixelSpectrum, ...]:
        """Return spectra for all current selections in display order."""
        return tuple(self.comparison_spectrum(item, mode=mode) for item in self._selections)

    def _catalog_maps_for_band(self, info: BandInfo) -> tuple[np.ndarray, np.ndarray]:
        if info.key in self._catalog_cache:
            return self._catalog_cache[info.key]
        if self._catalog_data is None or info.flux_column is None:
            raise RuntimeError("Catalogue-backed session has no flux column mapping.")
        if info.flux_column not in self._catalog_data.columns:
            raise ValueError(f"Catalog is missing drilling flux column '{info.flux_column}'.")

        science = np.full(self.shape, np.nan, dtype=float)
        uncertainty = np.full(self.shape, np.nan, dtype=float)
        error_values = (
            self._catalog_data[info.error_column]
            if info.error_column in self._catalog_data.columns
            else None
        )
        for row_index, identifier in enumerate(self._catalog_data["id"].astype(str)):
            pixel_index = self._catalog_pixel_index(identifier)
            if pixel_index is None:
                continue
            y_index, x_index = divmod(pixel_index, self.shape[1])
            if y_index >= self.shape[0]:
                continue
            science[y_index, x_index] = self._coerce_catalog_number(
                self._catalog_data.iloc[row_index][info.flux_column]
            )
            if error_values is not None:
                uncertainty[y_index, x_index] = self._coerce_catalog_number(
                    error_values.iloc[row_index]
                )
        self._catalog_cache[info.key] = (science, uncertainty)
        return self._catalog_cache[info.key]

    def _catalog_pixel_index(self, identifier: str) -> int | None:
        match = _PIXEL_ID_PATTERN.fullmatch(str(identifier))
        if match is None or match.group("galaxy") != self.galaxy:
            return None
        return int(match.group("pixel"))

    def _resolve_band(self, band: int | BandInfo | None) -> BandInfo:
        if band is None:
            return self.active_band
        if isinstance(band, BandInfo):
            if band not in self.bands:
                raise ValueError("BandInfo does not belong to this session.")
            return band
        if isinstance(band, (int, np.integer)):
            return self.bands[int(band)]
        raise TypeError("band must be None, an integer index, or BandInfo.")

    def _resolve_selection(self, selection: PixelSelection | int) -> PixelSelection:
        selection_id = selection.selection_id if isinstance(selection, PixelSelection) else int(selection)
        for candidate in self._selections:
            if candidate.selection_id == selection_id:
                return candidate
        raise KeyError(f"Selection {selection_id} is not present.")

    def _coerce_pixel(self, x: float, y: float) -> tuple[int, int]:
        try:
            x_float, y_float = float(x), float(y)
        except (TypeError, ValueError) as exc:
            raise TypeError("Pixel coordinates must be finite numbers.") from exc
        if not np.isfinite(x_float) or not np.isfinite(y_float):
            raise ValueError("Pixel coordinates must be finite numbers.")
        x_index, y_index = int(np.rint(x_float)), int(np.rint(y_float))
        if not (0 <= x_index < self.shape[1] and 0 <= y_index < self.shape[0]):
            raise IndexError(
                f"Pixel ({x_index}, {y_index}) is outside image bounds x=[0,{self.shape[1] - 1}], "
                f"y=[0,{self.shape[0] - 1}]."
            )
        return x_index, y_index

    def _sky_position(self, x: int, y: int) -> dict[str, float | None]:
        if self._reference_wcs is None:
            return {"ra_deg": None, "dec_deg": None}
        try:
            coordinate = self._reference_wcs.pixel_to_world(x, y)
            return {"ra_deg": float(coordinate.ra.deg), "dec_deg": float(coordinate.dec.deg)}
        except Exception:
            return {"ra_deg": None, "dec_deg": None}

    @staticmethod
    def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
        if len(shape) != 2:
            raise ValueError("A two-dimensional image shape is required.")
        result = tuple(int(value) for value in shape)
        if any(value <= 0 for value in result):
            raise ValueError("Image shape values must be positive.")
        return result

    @staticmethod
    def _sort_bands(bands: Iterable[BandInfo]) -> tuple[BandInfo, ...]:
        def sort_key(info: BandInfo):
            wavelength = info.pivot_wavelength
            known = wavelength is not None and np.isfinite(wavelength) and wavelength > 0
            return (0 if known else 1, float(wavelength) if known else np.inf, info.observatory.lower(), info.band.lower())

        return tuple(sorted(bands, key=sort_key))

    @staticmethod
    def _band_info_from_filter(image_set, *, observatory, band, flux_column, error_column, unit):
        properties: dict[str, float | None] = {
            "pivot_wavelength": None,
            "center_wavelength": None,
            "mean_wavelength": None,
            "fwhm": None,
        }
        try:
            curve = image_set.filter_inst.get_filter(name=band, facility=observatory)
        except Exception as exc:
            warnings.warn(
                f"Filter metadata is unavailable for {observatory}/{band}: {exc}. "
                "Its image remains available but has unknown wavelength.",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            for field in properties:
                try:
                    value = float(getattr(curve, field))
                except Exception:
                    continue
                properties[field] = value if np.isfinite(value) else None
        return BandInfo(
            observatory=observatory,
            band=band,
            flux_column=flux_column,
            error_column=error_column,
            unit=unit,
            **properties,
        )

    @staticmethod
    def _band_info_from_metadata(metadata: Mapping[str, Any], default_unit: str | None) -> BandInfo:
        def number(name: str) -> float | None:
            value = metadata.get(name)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        return BandInfo(
            observatory=str(metadata["observatory"]),
            band=str(metadata["band"]),
            flux_column=str(metadata["flux_column"]),
            error_column=str(metadata["error_column"]),
            pivot_wavelength=number("pivot_wavelength"),
            center_wavelength=number("center_wavelength"),
            mean_wavelength=number("mean_wavelength"),
            fwhm=number("fwhm"),
            unit=metadata.get("unit") or default_unit,
        )

    @staticmethod
    def _optional_wcs(header: fits.Header | None) -> WCS | None:
        if header is None:
            return None
        try:
            wcs = WCS(header, relax=True).celestial
        except Exception:
            return None
        if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
            return None
        return wcs

    @classmethod
    def _validate_wcs_alignment(
        cls,
        reference_header: fits.Header,
        headers: Mapping[tuple[str, str], fits.Header],
        shape: tuple[int, int],
        *,
        strict_wcs: bool,
        reference_key: tuple[str, str],
    ) -> None:
        if not strict_wcs:
            return
        reference = cls._optional_wcs(reference_header)
        if reference is None:
            raise ValueError("Reference image has no valid two-dimensional celestial WCS.")
        height, width = shape
        x = np.asarray([0.0, width - 1.0, 0.0, width - 1.0, (width - 1.0) / 2.0])
        y = np.asarray([0.0, 0.0, height - 1.0, height - 1.0, (height - 1.0) / 2.0])
        world = reference.pixel_to_world(x, y)
        for key, header in headers.items():
            if key == reference_key:
                continue
            candidate = cls._optional_wcs(header)
            if candidate is None:
                raise ValueError(f"{key[0]}/{key[1]} has no valid two-dimensional celestial WCS.")
            try:
                candidate_x, candidate_y = candidate.world_to_pixel(world)
                offset = np.hypot(candidate_x - x, candidate_y - y)
            except Exception as exc:
                raise ValueError(f"WCS alignment could not be checked for {key[0]}/{key[1]}: {exc}") from exc
            if not np.all(np.isfinite(offset)) or np.max(offset) > 1.0e-3:
                maximum = float(np.nanmax(offset)) if np.any(np.isfinite(offset)) else np.nan
                raise ValueError(
                    f"WCS misalignment for {key[0]}/{key[1]}: maximum offset is {maximum:.6g} pixels."
                )

    @staticmethod
    def _warn_if_psf_status_unknown(image_set, galaxy: str) -> None:
        psf_tree = getattr(image_set, "psf", {}).get(galaxy, {})
        values = [value for bands in psf_tree.values() if isinstance(bands, Mapping) for value in bands.values()]
        if not values or all(value in {None, ""} for value in values):
            warnings.warn(
                "PSF-matching status is not recorded for this image set; verify that all bands "
                "share an effective PSF before interpreting pixel SEDs physically.",
                RuntimeWarning,
                stacklevel=3,
            )

    @classmethod
    def _read_fits_catalog(cls, path: Path, galaxy: str | None) -> tuple[dict[str, Any], pd.DataFrame]:
        with fits.open(path, memmap=False) as hdul:
            if "DRILLMETA" not in hdul or "DRILLFILTERS" not in hdul:
                raise ValueError(
                    "Catalog has no drilling metadata. Recreate it with "
                    "write_catalog(include_drill_metadata=True)."
                )
            meta_frame = Table(hdul["DRILLMETA"].data).to_pandas()
            filter_frame = Table(hdul["DRILLFILTERS"].data).to_pandas()
            meta_frame = cls._decode_frame(meta_frame)
            filter_frame = cls._decode_frame(filter_frame)
            selected_galaxy = cls._select_catalog_galaxy(meta_frame, galaxy)
            row = meta_frame.loc[meta_frame["galaxy"].astype(str) == selected_galaxy].iloc[0]
            metadata = cls._metadata_from_fits_row(row, filter_frame, selected_galaxy)

            parts = []
            for hdu in hdul[1:]:
                if not isinstance(hdu, fits.BinTableHDU):
                    continue
                if str(hdu.name).upper() in {"DRILLMETA", "DRILLFILTERS"}:
                    continue
                if str(hdu.header.get("GALAXY", "")) != selected_galaxy:
                    continue
                parts.append(cls._decode_frame(Table(hdu.data).to_pandas()))
            if not parts:
                raise ValueError(f"FITS catalog has no data table for galaxy '{selected_galaxy}'.")
            data = cls._join_catalog_parts(parts)
        return metadata, data

    @classmethod
    def _read_ascii_catalog(
        cls,
        path: Path,
        galaxy: str | None,
        metadata_path: str | Path | None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        sidecar = Path(metadata_path) if metadata_path is not None else Path(f"{path}.spec7dt-drill.json")
        if not sidecar.exists():
            raise ValueError(
                f"Catalog metadata sidecar was not found: {sidecar}. Recreate it with "
                "write_catalog(include_drill_metadata=True) or pass metadata_path."
            )
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported or missing drilling metadata schema_version.")
        selected_galaxy = str(metadata.get("galaxy", ""))
        if not selected_galaxy:
            raise ValueError("Drilling metadata has no galaxy name.")
        if galaxy is not None and str(galaxy) != selected_galaxy:
            raise ValueError(f"Metadata is for '{selected_galaxy}', not requested galaxy '{galaxy}'.")
        try:
            data = Table.read(path).to_pandas()
        except Exception as exc:
            raise ValueError(f"Could not read catalog '{path}': {exc}") from exc
        return metadata, cls._decode_frame(data)

    @staticmethod
    def _decode_frame(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in result.columns:
            if result[column].dtype == object:
                result[column] = result[column].map(
                    lambda value: value.decode("utf-8").rstrip("\x00") if isinstance(value, bytes) else value
                )
        return result

    @staticmethod
    def _select_catalog_galaxy(meta_frame: pd.DataFrame, galaxy: str | None) -> str:
        galaxies = [str(value) for value in meta_frame["galaxy"].tolist()]
        if galaxy is None:
            if len(galaxies) != 1:
                raise ValueError("Catalog contains multiple galaxies; pass galaxy=<name>.")
            return galaxies[0]
        if str(galaxy) not in galaxies:
            raise ValueError(f"Galaxy '{galaxy}' is absent from drilling metadata.")
        return str(galaxy)

    @staticmethod
    def _metadata_from_fits_row(row: pd.Series, filters: pd.DataFrame, galaxy: str) -> dict[str, Any]:
        selected = filters.loc[filters["galaxy"].astype(str) == galaxy]
        return {
            "schema_version": int(row["schema_version"]),
            "galaxy": galaxy,
            "shape": [int(row["ny"]), int(row["nx"])],
            "flatten_order": str(row["flatten_order"]),
            "reference_header": str(row["reference_header"]),
            "flux_unit": str(row["flux_unit"]),
            "catalog_type": str(row["catalog_type"]),
            "invalid_values": json.loads(str(row["invalid_values"])),
            "filters": selected.drop(columns=["galaxy"]).to_dict(orient="records"),
        }

    @staticmethod
    def _join_catalog_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
        indexed = []
        for frame in parts:
            if "id" not in frame.columns:
                raise ValueError("FITS catalog table has no 'id' column.")
            indexed.append(frame.set_index("id", drop=True))
        joined = pd.concat(indexed, axis=1)
        joined = joined.loc[:, ~joined.columns.duplicated()]
        return joined.reset_index()

    @staticmethod
    def _header_from_string(value: str) -> fits.Header | None:
        if not value:
            return None
        try:
            return fits.Header.fromstring(value, sep="\n")
        except Exception:
            warnings.warn("Drilling metadata WCS header could not be read.", RuntimeWarning, stacklevel=3)
            return None

    @staticmethod
    def _coerce_catalog_number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan
