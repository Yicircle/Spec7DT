from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .metadata import GalaxyMetadataConfig, GalaxyMetadataResolver
from ..handlers.catalog_adapters import get_catalog_columns


_PIXEL_ID_PATTERN = re.compile(r"^(?P<galaxy>.+)_(?P<pixel>\d+)$")
_TOTAL_ID_PATTERN = re.compile(r"^.+_Total$")
_CATALOG_METADATA_COLUMNS = ("id", "redshift", "distance")
_FITS_MAX_COLUMNS = 999
_DRILL_METADATA_SCHEMA_VERSION = 1


class inputGenerator:
    def __init__(self):
        pass

    @classmethod
    def dataframe_generator(cls, image_set, cat_type, metadata_resolver=None):
        metadata_resolver = metadata_resolver or GalaxyMetadataResolver()
        galaxy_frames = []
        ordered_data_columns = []

        for galaxy, observatories in image_set.data.items():
            data_dict = {}
            expected_size = None

            for observatory, bands in observatories.items():
                for band, values in bands.items():
                    path = f"{galaxy}/{observatory}/{band}"
                    flat_values = np.asarray(values).flatten()
                    if expected_size is None:
                        expected_size = flat_values.size
                    elif flat_values.size != expected_size:
                        raise ValueError(
                            f"Image size mismatch within galaxy '{galaxy}': "
                            f"{path} has {flat_values.size} pixels; expected {expected_size}."
                        )

                    flat_error = np.asarray(
                        image_set.error[galaxy][observatory][band]
                    ).flatten()
                    if flat_error.size != flat_values.size:
                        raise ValueError(
                            f"Error-map size mismatch for {path}: "
                            f"{flat_error.size} pixels; expected {flat_values.size}."
                        )

                    flux_column = f"{observatory}.{band}"
                    error_column = f"{flux_column}_err"
                    data_dict[flux_column] = flat_values
                    data_dict[error_column] = flat_error
                    for column in (flux_column, error_column):
                        if column not in ordered_data_columns:
                            ordered_data_columns.append(column)

            if expected_size is None:
                continue

            local_df = pd.DataFrame({
                "id": [f"{galaxy}_{index}" for index in range(expected_size)],
                **data_dict,
            })

            for column in data_dict:
                if local_df[column].dtype.byteorder == ">":
                    dtype = local_df[column].dtype.newbyteorder("=")
                    local_df[column] = local_df[column].astype(dtype)
                local_df[column] = local_df[column].astype("float32")

            flux_columns = [column for column in data_dict if not column.endswith("_err")]
            if flux_columns:
                flux_cut = ((local_df[flux_columns] > 0).sum(axis=1)
                            >= (len(flux_columns) / 2))
                local_df = local_df.loc[flux_cut].copy()

                for flux_column in flux_columns:
                    error_column = f"{flux_column}_err"
                    mask = local_df[error_column] > 0.5 * local_df[flux_column]
                    local_df.loc[mask, [flux_column, error_column]] = np.nan

            galaxy_frames.append(local_df)

        if galaxy_frames:
            df = pd.concat(galaxy_frames, ignore_index=True, sort=False)
            df = df[["id", *ordered_data_columns]]
        else:
            df = pd.DataFrame(columns=["id", *ordered_data_columns])

        df = df.astype({"id": "str"})
        column_names = get_catalog_columns(cat_type, ordered_data_columns)
        df.rename(columns=column_names, inplace=True)

        # Consolidate the many flux/error blocks before adding metadata columns.
        df = df.copy()
        df["redshift"] = np.nan
        for galaxy in image_set.data:
            redshift = metadata_resolver.get_redshift(galaxy)
            if redshift is not None:
                mask = df["id"].str.startswith(f"{galaxy}_")
                df.loc[mask, "redshift"] = redshift

        renamed_data_columns = [column_names.get(column, column)
                                for column in ordered_data_columns]
        df = df[["id", *renamed_data_columns, "redshift"]]
        df.reset_index(drop=True, inplace=True)
        return df


def add_total_flux_rows(
    df: pd.DataFrame,
    galaxy_metadata: dict | None = None,
    *,
    metadata_config: GalaxyMetadataConfig | dict | None = None,
    invalid_values: tuple[float, ...] = (99.0, -99.0),
) -> pd.DataFrame:
    """Return a catalog copy with one integrated-flux row per galaxy.

    Pixel rows must use ``<galaxy>_<integer>`` IDs.  Manual metadata takes
    precedence over cached and live NED metadata.  Flux errors are combined
    in quadrature; other numeric catalog columns are summed.
    """
    if "id" not in df.columns:
        raise ValueError("add_total_flux_rows requires an 'id' column.")

    result = df.copy(deep=True)
    total_mask = result["id"].astype(str).str.match(_TOTAL_ID_PATTERN)
    result = result.loc[~total_mask].copy()

    galaxies = []
    row_galaxies = []
    invalid_ids = []
    for identifier in result["id"].astype(str):
        match = _PIXEL_ID_PATTERN.fullmatch(identifier)
        if match is None:
            invalid_ids.append(identifier)
            row_galaxies.append(None)
            continue
        galaxy = match.group("galaxy")
        row_galaxies.append(galaxy)
        if galaxy not in galaxies:
            galaxies.append(galaxy)

    if invalid_ids:
        examples = ", ".join(repr(identifier) for identifier in invalid_ids[:3])
        raise ValueError(
            "Pixel IDs must use '<galaxy>_<integer>' format; "
            f"invalid examples: {examples}."
        )
    if not galaxies:
        raise ValueError("No pixel rows are available for total-flux calculation.")

    result["redshift"] = np.nan
    result["distance"] = np.nan
    row_galaxies = pd.Series(row_galaxies, index=result.index, dtype="object")

    resolver = GalaxyMetadataResolver(
        metadata=galaxy_metadata,
        config=_total_metadata_config(metadata_config),
    )
    invalid_numbers = np.asarray(tuple(invalid_values), dtype=float)
    numeric_columns = [
        column for column in result.columns
        if column not in {"id", "redshift", "distance"}
        and pd.api.types.is_numeric_dtype(result[column])
    ]

    total_rows = []
    for galaxy in galaxies:
        galaxy_mask = row_galaxies == galaxy
        redshift = resolver.get_redshift(galaxy)
        distance = resolver.get_distance(galaxy)
        result.loc[galaxy_mask, "redshift"] = (
            np.nan if redshift is None else redshift
        )
        result.loc[galaxy_mask, "distance"] = (
            np.nan if distance is None else distance
        )

        total_row = {column: np.nan for column in result.columns}
        total_row["id"] = f"{galaxy}_Total"
        total_row["redshift"] = np.nan if redshift is None else redshift
        total_row["distance"] = np.nan if distance is None else distance

        for column in numeric_columns:
            values = pd.to_numeric(
                result.loc[galaxy_mask, column], errors="coerce"
            ).to_numpy(dtype=float)
            invalid = ~np.isfinite(values)
            if invalid_numbers.size:
                invalid |= np.isin(values, invalid_numbers)
            values[invalid] = np.nan
            if column.endswith("_err"):
                total_row[column] = float(np.sqrt(np.nansum(values ** 2)))
            else:
                total_row[column] = float(np.nansum(values))

        total_rows.append(total_row)

    totals_df = pd.DataFrame(total_rows, columns=result.columns)
    df_return = pd.concat([result, totals_df], ignore_index=True, sort=False)
    df_return = df_return.fillna(-99.0)
    return df_return


def _total_metadata_config(
    metadata_config: GalaxyMetadataConfig | dict | None,
) -> GalaxyMetadataConfig:
    if isinstance(metadata_config, GalaxyMetadataConfig):
        return replace(
            metadata_config,
            remote_sources=("ned",),
            fallback_order=("manual", "cache", "remote"),
        )

    options = dict(metadata_config or {})
    options["remote_sources"] = ("ned",)
    options["fallback_order"] = ("manual", "cache", "remote")
    return GalaxyMetadataConfig(**options)


def catalog_row_galaxies(df: pd.DataFrame) -> pd.Series:
    """Return the galaxy represented by every supported catalog row ID."""
    if "id" not in df.columns:
        raise ValueError("Catalog requires an 'id' column.")

    galaxies = []
    invalid_ids = []
    for identifier in df["id"].astype(str):
        pixel_match = _PIXEL_ID_PATTERN.fullmatch(identifier)
        if pixel_match is not None:
            galaxies.append(pixel_match.group("galaxy"))
            continue
        if _TOTAL_ID_PATTERN.fullmatch(identifier):
            galaxies.append(identifier.removesuffix("_Total"))
            continue
        galaxies.append(None)
        invalid_ids.append(identifier)

    if invalid_ids:
        examples = ", ".join(repr(identifier) for identifier in invalid_ids[:3])
        raise ValueError(
            "Catalog IDs must use '<galaxy>_<integer>' or '<galaxy>_Total' "
            f"format; invalid examples: {examples}."
        )
    return pd.Series(galaxies, index=df.index, dtype="object")


def galaxy_filter_columns(image_set, galaxy: str, catalog_type: str):
    """Resolve current image keys to their adapted catalog flux/error columns."""
    if galaxy not in image_set.data:
        available = ", ".join(sorted(image_set.data)) or "none"
        raise ValueError(
            f"Galaxy '{galaxy}' is not present in GalaxyImageSet. "
            f"Available galaxies: {available}."
        )

    raw_columns = []
    keys = []
    for observatory, bands in image_set.data[galaxy].items():
        for band in bands:
            flux_column = f"{observatory}.{band}"
            error_column = f"{flux_column}_err"
            raw_columns.extend((flux_column, error_column))
            keys.append((observatory, band, flux_column, error_column))

    mapping = get_catalog_columns(catalog_type, raw_columns)
    resolved = []
    owners = {}
    for observatory, band, raw_flux, raw_error in keys:
        flux_column = mapping.get(raw_flux, raw_flux)
        error_column = mapping.get(raw_error, raw_error)
        pair = (flux_column, error_column)
        owner = f"{observatory}/{band}"
        if pair in owners and owners[pair] != owner:
            raise ValueError(
                f"Catalog adapter maps both {owners[pair]} and {owner} to "
                f"the same columns {pair}."
            )
        owners[pair] = owner
        resolved.append((observatory, band, flux_column, error_column))
    return resolved


def prepare_galaxy_catalogs(
    image_set,
    catalog_frame,
    *,
    include_total=False,
    galaxy_metadata=None,
    metadata_config=None,
    invalid_values=(99.0, -99.0),
):
    """Build per-galaxy catalog views using the image set's real filters."""
    df = catalog_frame.data
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The stored pipeline catalog must be a pandas.DataFrame.")
    if not df.columns.is_unique:
        duplicates = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
        raise ValueError(f"Catalog contains duplicate columns: {duplicates}.")

    work = (
        add_total_flux_rows(
            df,
            galaxy_metadata,
            metadata_config=metadata_config,
            invalid_values=invalid_values,
        )
        if include_total
        else df.copy(deep=True)
    )
    row_galaxies = catalog_row_galaxies(work)
    catalog_galaxies = list(dict.fromkeys(row_galaxies.tolist()))
    missing_galaxies = sorted(set(catalog_galaxies) - set(image_set.data))
    if missing_galaxies:
        raise ValueError(
            "Catalog contains galaxies absent from GalaxyImageSet: "
            + ", ".join(missing_galaxies)
            + "."
        )

    result = {}
    filter_columns = {}
    for galaxy in sorted(catalog_galaxies):
        filters = galaxy_filter_columns(
            image_set,
            galaxy,
            catalog_frame.catalog_type,
        )
        missing_columns = sorted({
            column
            for _, _, flux_column, error_column in filters
            for column in (flux_column, error_column)
            if column not in work.columns
        })
        if missing_columns:
            raise ValueError(
                f"Catalog is missing expected columns for galaxy '{galaxy}': "
                + ", ".join(missing_columns)
                + "."
            )

        metadata_columns = [
            column for column in _CATALOG_METADATA_COLUMNS
            if column in work.columns
        ]
        measurement_columns = [
            column
            for _, _, flux_column, error_column in filters
            for column in (flux_column, error_column)
        ]
        result[galaxy] = (
            work.loc[row_galaxies == galaxy, metadata_columns + measurement_columns]
            .reset_index(drop=True)
            .copy()
        )
        filter_columns[galaxy] = filters

    if not result:
        raise ValueError("No galaxy rows are available in the stored catalog.")
    return result, filter_columns


def _normalize_catalog_format(output_path: Path, output_format):
    aliases = {
        "fits": "fits",
        "fit": "fits",
        "fts": "fits",
        "ascii": "ascii.ecsv",
        "ecsv": "ascii.ecsv",
        "ascii.ecsv": "ascii.ecsv",
        "csv": "ascii.csv",
        "ascii.csv": "ascii.csv",
        "txt": "ascii.basic",
        "dat": "ascii.basic",
        "ascii.basic": "ascii.basic",
    }
    if output_format is None:
        suffix = output_path.suffix.lower().lstrip(".")
        if suffix not in aliases:
            raise ValueError(
                "Cannot infer catalog format; use a .fits, .ecsv, .csv, .txt, "
                "or .dat suffix, or pass format explicitly."
            )
        return aliases[suffix]

    key = str(output_format).strip().lower()
    if key not in aliases:
        raise ValueError(
            f"Unsupported catalog format '{output_format}'. "
            "Use fits, ascii.ecsv, ascii.csv, or ascii.basic."
        )
    return aliases[key]


def _safe_output_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def _selected_white_bands(image_set, galaxy, bands):
    try:
        observatories = image_set.data[galaxy]
    except (AttributeError, KeyError, TypeError) as exc:
        raise KeyError(f"Galaxy '{galaxy}' is not present in GalaxyImageSet.") from exc

    available = [
        (observatory, band)
        for observatory, observatory_bands in observatories.items()
        for band in observatory_bands
    ]
    if bands is None:
        return sorted(available, key=lambda item: (str(item[0]), str(item[1])))

    selected = []
    for item in bands:
        if isinstance(item, str):
            raise TypeError("bands entries must be (observatory, band) pairs, not strings")
        try:
            observatory, band = item
        except (TypeError, ValueError) as exc:
            raise TypeError("bands entries must be (observatory, band) pairs") from exc
        key = (observatory, band)
        if key not in available:
            raise KeyError(f"Band '{observatory}/{band}' is not available for {galaxy}.")
        if key not in selected:
            selected.append(key)

    if not selected:
        raise ValueError("At least one band is required to build a white image.")
    return selected


def _white_image_unit(image_set, selected_headers):
    catalog_frame = getattr(image_set, "last_catalog", None)
    catalog_units = getattr(catalog_frame, "units", {}) if catalog_frame is not None else {}
    tracked_unit = catalog_units.get("flux") if hasattr(catalog_units, "get") else None
    if tracked_unit is not None and str(tracked_unit).strip().lower() != "native":
        return str(tracked_unit).strip()

    header_units = [str(header.get("BUNIT", "")).strip() for header in selected_headers]
    if header_units and all(header_units) and len(set(header_units)) == 1:
        return header_units[0]

    raise ValueError(
        "White-image bands must have a known common flux unit. Run the pipeline "
        "with unit conversion enabled or provide consistent BUNIT headers."
    )


def _white_image_wcs(header, shape, context):
    try:
        normalized_shape = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: a positive 2-D image shape is required") from exc
    if (
        len(normalized_shape) != 2
        or any(value <= 0 for value in normalized_shape)
        or normalized_shape != tuple(shape)
    ):
        raise ValueError(f"{context}: a positive 2-D image shape is required")
    try:
        wcs = WCS(header, relax=True).celestial
    except Exception as exc:
        raise ValueError(f"{context}: celestial WCS could not be parsed ({exc})") from exc
    if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
        raise ValueError(f"{context}: a two-dimensional celestial WCS is required")
    return wcs


def _validate_white_image_alignment(reference_wcs, candidate_wcs, shape, context):
    height, width = shape
    x = np.asarray([0.0, width - 1.0, 0.0, width - 1.0, (width - 1.0) / 2.0])
    y = np.asarray([0.0, 0.0, height - 1.0, height - 1.0, (height - 1.0) / 2.0])
    try:
        world = reference_wcs.pixel_to_world(x, y)
        candidate_x, candidate_y = candidate_wcs.world_to_pixel(world)
        offset = np.hypot(candidate_x - x, candidate_y - y)
    except Exception as exc:
        raise ValueError(f"{context}: WCS alignment could not be checked ({exc})") from exc
    if not np.all(np.isfinite(offset)) or np.max(offset) > 1.0e-3:
        maximum = float(np.nanmax(offset)) if np.any(np.isfinite(offset)) else np.nan
        raise ValueError(
            f"{context}: WCS differs from the reference grid by up to {maximum:.6g} pixels"
        )


def write_white_image(
    image_set,
    output_path,
    *,
    galaxy,
    bands=None,
    overwrite=False,
):
    """Write a final-grid inverse-variance weighted diagnostic image.

    ``bands`` is an optional iterable of ``(observatory, band)`` pairs. Error
    arrays are interpreted as per-pixel standard deviations. The input image
    set is never modified.
    """
    selected = _selected_white_bands(image_set, galaxy, bands)
    if not selected:
        raise ValueError(f"No bands are available for {galaxy}.")

    entries = []
    for observatory, band in selected:
        context = f"{galaxy}/{observatory}/{band}"
        try:
            data = np.asarray(image_set.data[galaxy][observatory][band])
            error = image_set.error[galaxy][observatory][band]
            header = image_set.header[galaxy][observatory][band]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(f"White-image inputs are incomplete for {context}.") from exc
        if error is None:
            raise ValueError(f"White-image error map is missing for {context}.")
        error = np.asarray(error)
        if data.ndim != 2 or error.shape != data.shape:
            raise ValueError(
                f"White-image science/error shapes are invalid for {context}: "
                f"science={data.shape}, error={error.shape}."
            )
        entries.append((observatory, band, data, error, header, context))

    reference_shape = entries[0][2].shape
    reference_header = entries[0][4]
    reference_wcs = _white_image_wcs(reference_header, reference_shape, entries[0][5])
    for _, _, data, _, header, context in entries[1:]:
        if data.shape != reference_shape:
            raise ValueError(
                f"White-image bands must share one final shape: reference={reference_shape}, "
                f"{context}={data.shape}."
            )
        candidate_wcs = _white_image_wcs(header, data.shape, context)
        _validate_white_image_alignment(reference_wcs, candidate_wcs, reference_shape, context)

    common_unit = _white_image_unit(image_set, [entry[4] for entry in entries])
    weighted_sum = np.zeros(reference_shape, dtype=np.float64)
    inverse_variance = np.zeros(reference_shape, dtype=np.float64)
    contributor_count = np.zeros(reference_shape, dtype=np.uint16)
    band_rows = []

    for observatory, band, data, error, _, context in entries:
        data = np.asarray(data, dtype=np.float64)
        error = np.asarray(error, dtype=np.float64)
        valid = np.isfinite(data) & np.isfinite(error) & (error > 0)
        weight = np.zeros(reference_shape, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(1.0, error * error, out=weight, where=valid)
        valid &= np.isfinite(weight) & (weight > 0)
        weight[~valid] = 0.0
        weighted_sum[valid] += weight[valid] * data[valid]
        inverse_variance += weight
        contributor_count[valid] += 1
        error_source = (
            getattr(image_set, "_error_source", {})
            .get(galaxy, {})
            .get(observatory, {})
            .get(band, "provided")
        )
        band_rows.append((str(observatory), str(band), str(error_source), int(np.sum(valid))))

    covered = inverse_variance > 0
    if not np.any(covered):
        raise ValueError(f"No valid inverse-variance samples are available for {galaxy}.")
    white = np.full(reference_shape, np.nan, dtype=np.float64)
    np.divide(weighted_sum, inverse_variance, out=white, where=covered)

    output_header = reference_wcs.to_header(relax=True)
    output_header["OBJECT"] = str(galaxy)
    output_header["CREATOR"] = "Spec7DT"
    output_header["PRODTYPE"] = "WHITE"
    output_header["WHTMETH"] = "IVAR"
    output_header["NBANDS"] = len(entries)
    output_header["REFBAND"] = f"{entries[0][0]}/{entries[0][1]}"
    output_header["BUNIT"] = common_unit
    output_header["NVALID"] = int(np.sum(covered))
    output_header["COVFRAC"] = float(np.mean(covered))
    finite_white = white[covered]
    output_header["DATAMIN"] = float(np.min(finite_white))
    output_header["DATAMAX"] = float(np.max(finite_white))
    output_header["DATAMED"] = float(np.median(finite_white))
    output_header.add_history("Inverse-variance weighted diagnostic image; not a photometric band.")

    inverse_header = output_header.copy()
    inverse_header["BUNIT"] = f"({common_unit})^-2"
    count_header = output_header.copy()
    count_header["BUNIT"] = "count"
    band_table = Table(
        rows=band_rows,
        names=("observatory", "band", "error_source", "valid_pixels"),
        dtype=("U64", "U64", "U32", "i8"),
    )
    bands_hdu = fits.table_to_hdu(band_table)
    bands_hdu.name = "BANDS"

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".fits", ".fit", ".fts"}:
        output_path = output_path.with_suffix(".fits")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([
        fits.PrimaryHDU(np.asarray(white, dtype=np.float32), header=output_header),
        fits.ImageHDU(
            np.asarray(inverse_variance, dtype=np.float32),
            header=inverse_header,
            name="IVAR",
        ),
        fits.ImageHDU(contributor_count, header=count_header, name="NCONTRIB"),
        bands_hdu,
    ]).writeto(output_path, overwrite=overwrite)
    return output_path


def _drilling_number(value):
    """Return a JSON-safe finite float or ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _drilling_header_string(header):
    try:
        return header.tostring(sep="\n", endcard=True, padding=False)
    except (AttributeError, TypeError):
        return ""


def _build_drilling_metadata(
    image_set,
    catalog_frame,
    filter_columns,
    *,
    invalid_values,
):
    """Build serialisable 2-D catalogue reconstruction metadata by galaxy."""
    metadata_by_galaxy = {}
    catalog_units = getattr(catalog_frame, "units", {})
    catalog_flux_unit = (
        catalog_units.get("flux")
        if hasattr(catalog_units, "get")
        else None
    )

    for galaxy, filters in filter_columns.items():
        if not filters:
            raise ValueError(f"No filters are available for drilling metadata for {galaxy}.")
        reference_observatory, reference_band, _, _ = filters[0]
        reference_data = np.asarray(
            image_set.data[galaxy][reference_observatory][reference_band]
        )
        if reference_data.ndim != 2 or reference_data.size == 0:
            raise ValueError(
                f"Drilling metadata requires a non-empty 2-D reference image for "
                f"{galaxy}/{reference_observatory}/{reference_band}."
            )
        reference_header = image_set.header[galaxy][reference_observatory][reference_band]
        flux_unit = (
            str(catalog_flux_unit)
            if catalog_flux_unit is not None
            else str(reference_header.get("BUNIT", "native"))
        )

        filter_metadata = []
        for observatory, band, flux_column, error_column in filters:
            data = np.asarray(image_set.data[galaxy][observatory][band])
            if data.shape != reference_data.shape:
                raise ValueError(
                    f"Drilling metadata requires aligned shapes for {galaxy}: "
                    f"reference={reference_data.shape}, {observatory}/{band}={data.shape}."
                )
            wavelengths = {
                "pivot_wavelength": None,
                "center_wavelength": None,
                "mean_wavelength": None,
                "fwhm": None,
            }
            try:
                curve = image_set.filter_inst.get_filter(name=band, facility=observatory)
            except Exception:
                pass
            else:
                for name in wavelengths:
                    try:
                        wavelengths[name] = _drilling_number(getattr(curve, name))
                    except Exception:
                        continue
            filter_metadata.append({
                "observatory": str(observatory),
                "band": str(band),
                "flux_column": str(flux_column),
                "error_column": str(error_column),
                "unit": flux_unit,
                **wavelengths,
            })

        metadata_by_galaxy[str(galaxy)] = {
            "schema_version": _DRILL_METADATA_SCHEMA_VERSION,
            "galaxy": str(galaxy),
            "shape": [int(reference_data.shape[0]), int(reference_data.shape[1])],
            "flatten_order": "C",
            "reference_observatory": str(reference_observatory),
            "reference_band": str(reference_band),
            "reference_header": _drilling_header_string(reference_header),
            "flux_unit": flux_unit,
            "catalog_type": str(getattr(catalog_frame, "catalog_type", "")),
            "invalid_values": [float(value) for value in invalid_values],
            "filters": filter_metadata,
        }
    return metadata_by_galaxy


def _drilling_metadata_hdus(metadata_by_galaxy):
    """Convert drilling metadata to self-contained FITS table extensions."""
    meta_rows = []
    filter_rows = []
    for galaxy, metadata in metadata_by_galaxy.items():
        meta_rows.append({
            "schema_version": int(metadata["schema_version"]),
            "galaxy": galaxy,
            "ny": int(metadata["shape"][0]),
            "nx": int(metadata["shape"][1]),
            "flatten_order": metadata["flatten_order"],
            "reference_observatory": metadata["reference_observatory"],
            "reference_band": metadata["reference_band"],
            "reference_header": metadata["reference_header"],
            "flux_unit": metadata["flux_unit"],
            "catalog_type": metadata["catalog_type"],
            "invalid_values": json.dumps(metadata["invalid_values"]),
        })
        for filter_metadata in metadata["filters"]:
            filter_rows.append({
                "galaxy": galaxy,
                "observatory": filter_metadata["observatory"],
                "band": filter_metadata["band"],
                "flux_column": filter_metadata["flux_column"],
                "error_column": filter_metadata["error_column"],
                "unit": filter_metadata["unit"],
                "pivot_wavelength": (
                    np.nan
                    if filter_metadata["pivot_wavelength"] is None
                    else filter_metadata["pivot_wavelength"]
                ),
                "center_wavelength": (
                    np.nan
                    if filter_metadata["center_wavelength"] is None
                    else filter_metadata["center_wavelength"]
                ),
                "mean_wavelength": (
                    np.nan
                    if filter_metadata["mean_wavelength"] is None
                    else filter_metadata["mean_wavelength"]
                ),
                "fwhm": np.nan if filter_metadata["fwhm"] is None else filter_metadata["fwhm"],
            })

    metadata_hdu = fits.table_to_hdu(Table(rows=meta_rows))
    metadata_hdu.name = "DRILLMETA"
    filters_hdu = fits.table_to_hdu(Table(rows=filter_rows))
    filters_hdu.name = "DRILLFILTERS"
    return [metadata_hdu, filters_hdu]


def write_image_set_catalog(
    image_set,
    catalog_frame,
    output_path,
    *,
    format=None,
    include_total=False,
    galaxy_metadata=None,
    metadata_config=None,
    invalid_values=(99.0, -99.0),
    include_drill_metadata=False,
    overwrite=False,
):
    """Write a stored pipeline catalog using per-galaxy filter selections."""
    output_path = Path(output_path)
    output_format = _normalize_catalog_format(output_path, format)
    galaxy_catalogs, filter_columns = prepare_galaxy_catalogs(
        image_set,
        catalog_frame,
        include_total=include_total,
        galaxy_metadata=galaxy_metadata,
        metadata_config=metadata_config,
        invalid_values=invalid_values,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drilling_metadata = (
        _build_drilling_metadata(
            image_set,
            catalog_frame,
            filter_columns,
            invalid_values=invalid_values,
        )
        if include_drill_metadata
        else None
    )

    if output_format == "fits":
        if output_path.suffix.lower() not in {".fits", ".fit", ".fts"}:
            output_path = output_path.with_suffix(".fits")

        hdus = [fits.PrimaryHDU()]
        hdus[0].header["CREATOR"] = "Spec7DT"
        for galaxy, galaxy_df in galaxy_catalogs.items():
            metadata_columns = [
                column for column in _CATALOG_METADATA_COLUMNS
                if column in galaxy_df.columns
            ]
            filters = filter_columns[galaxy]
            pair_capacity = (_FITS_MAX_COLUMNS - len(metadata_columns)) // 2
            if pair_capacity < 1:
                raise ValueError(
                    f"Too many metadata columns to create a FITS table for {galaxy}."
                )
            chunks = [
                filters[index:index + pair_capacity]
                for index in range(0, len(filters), pair_capacity)
            ] or [[]]
            safe_galaxy = _safe_output_component(galaxy).upper()

            for part_index, filter_chunk in enumerate(chunks, start=1):
                measurement_columns = [
                    column
                    for _, _, flux_column, error_column in filter_chunk
                    for column in (flux_column, error_column)
                ]
                part_df = galaxy_df[metadata_columns + measurement_columns]
                hdu = fits.table_to_hdu(Table.from_pandas(part_df, index=False))
                hdu.name = (
                    safe_galaxy
                    if len(chunks) == 1
                    else f"{safe_galaxy}_P{part_index:02d}"
                )
                hdu.header["GALAXY"] = str(galaxy)
                hdu.header["PART"] = part_index
                hdu.header["NPART"] = len(chunks)
                hdus.append(hdu)

        if drilling_metadata is not None:
            hdus.extend(_drilling_metadata_hdus(drilling_metadata))

        fits.HDUList(hdus).writeto(output_path, overwrite=overwrite)
        return [output_path]

    suffix_by_format = {
        "ascii.ecsv": ".ecsv",
        "ascii.csv": ".csv",
        "ascii.basic": ".txt",
    }
    suffix = output_path.suffix or suffix_by_format[output_format]
    catalog_items = list(galaxy_catalogs.items())
    targets = [
        output_path.with_name(
            f"{output_path.stem}_{_safe_output_component(galaxy)}{suffix}"
        )
        for galaxy, _ in catalog_items
    ]
    sidecars = [Path(f"{target}.spec7dt-drill.json") for target in targets]
    write_targets = targets + (sidecars if drilling_metadata is not None else [])
    existing = [path for path in write_targets if path.exists()]
    if existing and not overwrite:
        raise OSError(
            "Catalog output already exists: "
            + ", ".join(str(path) for path in existing)
            + "."
        )

    for (galaxy, galaxy_df), target, sidecar in zip(catalog_items, targets, sidecars):
        Table.from_pandas(galaxy_df, index=False).write(
            target,
            format=output_format,
            overwrite=overwrite,
        )
        if drilling_metadata is not None:
            sidecar.write_text(
                json.dumps(drilling_metadata[galaxy], indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return targets
