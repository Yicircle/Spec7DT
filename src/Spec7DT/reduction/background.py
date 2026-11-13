from dataclasses import dataclass
from time import perf_counter
from typing import Literal
import warnings

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources
from scipy.ndimage import binary_dilation

from ..core import AnalysisProduct
from ..utils.reporting import emit_detail
from ..utils.utility import useful_functions


_THUMBNAIL_MAX_AXIS = 1024
_AUTO_MESH_COUNTS = (8, 12, 16)
_AUTO_FILTER_SIZES = ((3, 3), (1, 1))
_VALIDATION_GRID = 8
_MIN_SKY_FRACTION = 0.30
_MIN_VALIDATION_CELLS = 16
_SOURCE_SIGMA = 2.5
_SOURCE_DILATION = 3
_RESIDUAL_TARGET_SCALE = 1.25


@dataclass(frozen=True)
class BackgroundConfig:
    """Background subtraction and missing-error estimation settings.

    ``mode='auto'`` selects the simplest model that leaves an acceptably flat
    source-masked sky. The numerical search settings are deliberately private,
    so the common pipeline path requires no background tuning.
    """

    error_mode: Literal["local_rms", "global_rms"] = "local_rms"
    fallback: Literal["global_rms", "error"] = "global_rms"
    mode: Literal["auto", "constant", "legacy"] = "auto"
    post_residual: bool = True

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("background_config must be a BackgroundConfig, dict, or None")

    def __post_init__(self):
        if self.error_mode not in {"local_rms", "global_rms"}:
            raise ValueError("error_mode must be 'local_rms' or 'global_rms'")
        if self.fallback not in {"global_rms", "error"}:
            raise ValueError("fallback must be 'global_rms' or 'error'")
        if self.mode not in {"auto", "constant", "legacy"}:
            raise ValueError("mode must be 'auto', 'constant', or 'legacy'")
        if not isinstance(self.post_residual, bool):
            raise TypeError("post_residual must be bool")


class _BackgroundEstimate:
    """Background/RMS provider plus compact audit diagnostics."""

    def __init__(self, background_source, rms_source=None, diagnostics=None):
        self._background_source = background_source
        self._rms_source = rms_source if rms_source is not None else background_source
        self.diagnostics = dict(diagnostics or {})

    @staticmethod
    def _value(source, attribute):
        try:
            return getattr(source, attribute)
        except AttributeError:
            return source

    @property
    def background(self):
        return self._value(self._background_source, "background")

    @property
    def background_rms(self):
        return self._value(self._rms_source, "background_rms")

    @property
    def background_rms_median(self):
        if hasattr(self._rms_source, "background_rms_median"):
            return float(self._rms_source.background_rms_median)
        values = np.asarray(self.background_rms, dtype=float)
        valid = values[np.isfinite(values) & (values > 0)]
        return float(np.nanmedian(valid)) if valid.size else np.nan


class _ScalarRms:
    def __init__(self, shape, value):
        self.shape = tuple(shape)
        self.background_rms_median = float(value)

    @property
    def background_rms(self):
        return np.full(self.shape, self.background_rms_median, dtype=np.float32)


def _target_name(galaxy_name, observatory, band):
    return f"{galaxy_name}/{observatory}/{band}"


def _target_path(image_set, galaxy_name, observatory, band):
    try:
        return image_set.files[galaxy_name][observatory][band]
    except (AttributeError, KeyError, TypeError):
        return _target_name(galaxy_name, observatory, band)


def _error_source(image_set, error_data, galaxy_name, observatory, band):
    getter = getattr(image_set, "get_error_source", None)
    if callable(getter):
        return getter(galaxy_name, observatory, band)
    return "missing" if error_data is None else "provided"


def has_missing_error_maps(image_set):
    checker = getattr(image_set, "has_missing_errors", None)
    if callable(checker):
        return checker()

    error_tree = getattr(image_set, "error", {})
    for galaxy_name, observatories in getattr(image_set, "data", {}).items():
        for observatory, bands in observatories.items():
            for band in bands:
                try:
                    error = error_tree[galaxy_name][observatory][band]
                except (KeyError, TypeError):
                    return True
                if error is None:
                    return True
    return False


def _coverage_mask(image_data):
    image = np.asarray(image_data)
    return ~np.isfinite(image) | (image == 0)


def _legacy_background_parameters(shape):
    ny, nx = shape
    box_size = (max(1, int(ny * 0.25)), max(1, int(nx * 0.25)))
    return box_size, (3, 3)


def _mesh_box_size(shape, mesh_count):
    return tuple(max(1, int(np.ceil(size / float(mesh_count)))) for size in shape)


def _background2d(data, coverage_mask, *, box_size, filter_size, source_mask=None):
    return Background2D(
        data,
        box_size=box_size,
        filter_size=filter_size,
        mask=source_mask,
        coverage_mask=coverage_mask,
        fill_value=0.0,
        sigma_clip=SigmaClip(sigma=3.0),
        bkg_estimator=MedianBackground(),
        exclude_percentile=50.0,
    )


def _robust_location_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    _, median, rms = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    if not np.isfinite(rms) or rms <= 0:
        median = float(np.nanmedian(values))
        rms = float(1.4826 * np.nanmedian(np.abs(values - median)))
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.nanstd(values))
    return float(median), float(rms)


def _thumbnail(image_data):
    image = np.asarray(image_data)
    stride = max(1, int(np.ceil(max(image.shape) / _THUMBNAIL_MAX_AXIS)))
    return np.asarray(image[::stride, ::stride], dtype=np.float32), stride


def _source_mask(image, coverage_mask, median, rms):
    mask = np.zeros(image.shape, dtype=bool)
    if not np.isfinite(rms) or rms <= 0:
        return mask
    residual = np.asarray(image, dtype=float) - float(median)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            segments = detect_sources(
                residual,
                threshold=float(_SOURCE_SIGMA * rms),
                npixels=5,
                mask=coverage_mask,
            )
    except Exception:
        segments = None
    if segments is not None:
        mask = np.asarray(segments.data > 0, dtype=bool)
        mask = binary_dilation(mask, iterations=_SOURCE_DILATION)
    mask[coverage_mask] = False
    return mask


def _fit_plane(data, valid):
    data = np.asarray(data)
    valid = np.asarray(valid, dtype=bool)
    if data.size <= _THUMBNAIL_MAX_AXIS**2:
        y_grid, x_grid = np.indices(data.shape, dtype=np.float32)
        x_norm = 2.0 * x_grid / max(1.0, data.shape[1] - 1.0) - 1.0
        y_norm = 2.0 * y_grid / max(1.0, data.shape[0] - 1.0) - 1.0
        use = valid.copy()
        coefficients = None
        for _ in range(5):
            use_count = np.count_nonzero(use)
            if use_count < 3:
                break
            design = np.column_stack(
                (
                    np.ones(use_count, dtype=np.float32),
                    x_norm[use],
                    y_norm[use],
                )
            )
            coefficients = np.linalg.lstsq(design, data[use], rcond=None)[0]
            model = (
                coefficients[0]
                + coefficients[1] * x_norm
                + coefficients[2] * y_norm
            )
            residual = np.asarray(data, dtype=float) - model
            center, rms = _robust_location_scale(residual[use])
            if not np.isfinite(rms) or rms <= 0:
                break
            updated = valid & (np.abs(residual - center) <= 3.0 * rms)
            if np.array_equal(updated, use):
                break
            use = updated
        if coefficients is None:
            raise ValueError("Insufficient sky pixels for a background plane")
        return np.asarray(model, dtype=np.float32), tuple(
            float(value) for value in coefficients
        )

    use = np.asarray(valid, dtype=bool).copy()
    x_norm_full = (
        2.0 * np.arange(data.shape[1], dtype=float)
        / max(1.0, data.shape[1] - 1.0)
        - 1.0
    )
    x_norm_squared = x_norm_full**2
    coefficients = None
    for _ in range(5):
        use_count = np.count_nonzero(use)
        if use_count < 3:
            break

        normal = np.zeros((3, 3), dtype=float)
        right = np.zeros(3, dtype=float)
        for y0 in range(0, data.shape[0], 256):
            y1 = min(data.shape[0], y0 + 256)
            block_use = use[y0:y1]
            if not np.any(block_use):
                continue
            y_norm = (
                2.0 * np.arange(y0, y1, dtype=float)
                / max(1.0, data.shape[0] - 1.0)
                - 1.0
            )
            column_counts = np.sum(block_use, axis=0, dtype=float)
            row_counts = np.sum(block_use, axis=1, dtype=float)
            count = float(np.sum(row_counts))
            sum_x = np.dot(column_counts, x_norm_full)
            sum_y = np.dot(row_counts, y_norm)
            sum_xy = np.dot(y_norm, block_use @ x_norm_full)
            normal += np.array(
                [
                    [count, sum_x, sum_y],
                    [sum_x, np.dot(column_counts, x_norm_squared), sum_xy],
                    [sum_y, sum_xy, np.dot(row_counts, y_norm**2)],
                ]
            )
            values = np.where(block_use, data[y0:y1], 0.0)
            column_sums = np.sum(values, axis=0, dtype=float)
            row_sums = np.sum(values, axis=1, dtype=float)
            right += np.array(
                [
                    np.sum(row_sums),
                    np.dot(column_sums, x_norm_full),
                    np.dot(row_sums, y_norm),
                ]
            )
        coefficients = np.linalg.lstsq(normal, right, rcond=None)[0]

        residual_values = np.empty(use_count, dtype=float)
        offset = 0
        for y0 in range(0, data.shape[0], 256):
            y1 = min(data.shape[0], y0 + 256)
            block_use = use[y0:y1]
            count = np.count_nonzero(block_use)
            if count == 0:
                continue
            y_norm = (
                2.0 * np.arange(y0, y1, dtype=float)
                / max(1.0, data.shape[0] - 1.0)
                - 1.0
            )
            model = (
                coefficients[0]
                + coefficients[1] * x_norm_full[None, :]
                + coefficients[2] * y_norm[:, None]
            )
            residual = np.asarray(data[y0:y1], dtype=float) - model
            residual_values[offset : offset + count] = residual[block_use]
            offset += count
        center, rms = _robust_location_scale(residual_values[:offset])
        numerical_floor = (
            np.finfo(data.dtype).eps
            if np.issubdtype(data.dtype, np.floating)
            else np.finfo(float).eps
        ) * max(1.0, float(np.sum(np.abs(coefficients)))) * 50.0
        if not np.isfinite(rms) or rms <= numerical_floor:
            break

        updated = np.zeros(valid.shape, dtype=bool)
        for y0 in range(0, data.shape[0], 256):
            y1 = min(data.shape[0], y0 + 256)
            block_valid = valid[y0:y1]
            if not np.any(block_valid):
                continue
            y_norm = (
                2.0 * np.arange(y0, y1, dtype=float)
                / max(1.0, data.shape[0] - 1.0)
                - 1.0
            )
            model = (
                coefficients[0]
                + coefficients[1] * x_norm_full[None, :]
                + coefficients[2] * y_norm[:, None]
            )
            residual = np.asarray(data[y0:y1], dtype=float) - model
            updated[y0:y1] = block_valid & (
                np.abs(residual - center) <= 3.0 * rms
            )
        if np.array_equal(updated, use):
            break
        use = updated
    if coefficients is None:
        raise ValueError("Insufficient sky pixels for a background plane")
    model = _plane_from_coefficients(data.shape, coefficients)
    return model, tuple(float(v) for v in coefficients)


def _validation_metrics(residual, valid, noise_rms):
    residual = np.asarray(residual, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(residual)
    median, measured_rms = _robust_location_scale(residual[valid])
    scale = float(noise_rms)
    if not np.isfinite(scale) or scale <= 0:
        scale = measured_rms
    if not np.isfinite(scale) or scale <= 0:
        scale = np.finfo(float).eps

    cell_medians = []
    y_edges = np.linspace(0, residual.shape[0], _VALIDATION_GRID + 1, dtype=int)
    x_edges = np.linspace(0, residual.shape[1], _VALIDATION_GRID + 1, dtype=int)
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
            cell_valid = valid[y0:y1, x0:x1]
            minimum = max(10, int(np.ceil(cell_valid.size * 0.30)))
            if np.count_nonzero(cell_valid) < minimum:
                continue
            cell = residual[y0:y1, x0:x1]
            cell_median, _ = _robust_location_scale(cell[cell_valid])
            if np.isfinite(cell_median):
                cell_medians.append(cell_median)

    q90 = (
        float(np.percentile(np.abs(cell_medians), 90))
        if cell_medians
        else np.inf
    )
    _, cell_rms = _robust_location_scale(cell_medians)
    passes = (
        len(cell_medians) >= _MIN_VALIDATION_CELLS
        and np.isfinite(median)
        and abs(median) <= 0.1 * scale
        and q90 <= 0.5 * scale
    )
    rejection_reasons = []
    if len(cell_medians) < _MIN_VALIDATION_CELLS:
        rejection_reasons.append(
            f"validation_cells={len(cell_medians)}<{_MIN_VALIDATION_CELLS}"
        )
    if not np.isfinite(median) or abs(median) > 0.1 * scale:
        rejection_reasons.append("global_median>0.1*rms")
    if not np.isfinite(q90) or q90 > 0.5 * scale:
        rejection_reasons.append("cell_q90>0.5*rms")
    return {
        "residual_median": float(median),
        "residual_rms": float(measured_rms),
        "cell_q90": q90,
        "cell_rms": float(cell_rms),
        "valid_cells": int(len(cell_medians)),
        "passes": bool(passes),
        "score": float(q90 / scale + 1.0e-3 * abs(median) / scale),
        "rejection_reason": "; ".join(rejection_reasons),
    }


def _candidate_record(algorithm, background, image, sky_valid, noise_rms, **extra):
    metrics = _validation_metrics(np.asarray(image) - background, sky_valid, noise_rms)
    return {
        "algorithm": algorithm,
        "background": background,
        **metrics,
        **extra,
    }


def _constant_candidate(image, sky_valid, noise_rms):
    value, _ = _robust_location_scale(np.asarray(image)[sky_valid])
    if not np.isfinite(value):
        value, _ = _robust_location_scale(np.asarray(image)[np.isfinite(image)])
    background = np.full(image.shape, value, dtype=np.float32)
    return _candidate_record(
        "constant",
        background,
        image,
        sky_valid,
        noise_rms,
        value=float(value),
        box_size=None,
        filter_size=None,
        coefficients=(float(value), 0.0, 0.0),
    )


def _automatic_thumbnail_model(image, coverage_mask, source_mask, noise_rms, mode):
    sky_valid = ~coverage_mask & ~source_mask
    sky_count = np.count_nonzero(sky_valid)
    sky_fraction = sky_count / max(1, np.count_nonzero(~coverage_mask))
    constant = _constant_candidate(image, sky_valid, noise_rms)
    constant["sky_fraction"] = float(sky_fraction)
    constant["model_uncertainty"] = float(
        noise_rms / np.sqrt(sky_count)
        if sky_count and np.isfinite(noise_rms)
        else np.nan
    )
    candidates = [constant]

    if mode == "constant" or sky_fraction < _MIN_SKY_FRACTION:
        constant["fallback"] = sky_fraction < _MIN_SKY_FRACTION
        if sky_fraction < _MIN_SKY_FRACTION:
            constant["fallback_reason"] = (
                "no unmasked sky" if sky_count == 0 else "usable sky below 30%"
            )
        return constant, candidates

    try:
        plane, coefficients = _fit_plane(image, sky_valid)
        candidates.append(
            _candidate_record(
                "plane",
                plane,
                image,
                sky_valid,
                noise_rms,
                coefficients=coefficients,
                model_uncertainty=float(
                    noise_rms * np.sqrt(3.0 / sky_count)
                    if sky_count and np.isfinite(noise_rms)
                    else np.nan
                ),
                box_size=None,
                filter_size=None,
                sky_fraction=float(sky_fraction),
            )
        )
    except ValueError:
        pass

    for mesh_count in _AUTO_MESH_COUNTS:
        box_size = _mesh_box_size(image.shape, mesh_count)
        for filter_size in _AUTO_FILTER_SIZES:
            try:
                model = _background2d(
                    image,
                    coverage_mask,
                    box_size=box_size,
                    filter_size=filter_size,
                    source_mask=source_mask,
                )
                background = np.asarray(model.background, dtype=np.float32)
            except Exception:
                continue
            candidates.append(
                _candidate_record(
                    "mesh",
                    background,
                    image,
                    sky_valid,
                    noise_rms,
                    mesh_count=int(mesh_count),
                    box_size=tuple(int(v) for v in box_size),
                    filter_size=tuple(int(v) for v in filter_size),
                    coefficients=None,
                    model_uncertainty=float(
                        noise_rms / np.sqrt(sky_count)
                        if sky_count and np.isfinite(noise_rms)
                        else np.nan
                    ),
                    sky_fraction=float(sky_fraction),
                )
            )

    selected = next((candidate for candidate in candidates if candidate["passes"]), None)
    if selected is None:
        finite = [candidate for candidate in candidates if np.isfinite(candidate["score"])]
        selected = min(finite, key=lambda item: item["score"]) if finite else constant
    return selected, candidates


def _expanded_mask(mask, stride, shape):
    if stride == 1:
        return np.asarray(mask, dtype=bool)
    expanded = np.repeat(np.repeat(mask, stride, axis=0), stride, axis=1)
    return np.asarray(expanded[: shape[0], : shape[1]], dtype=bool)


def _plane_from_coefficients(shape, coefficients):
    intercept, x_coefficient, y_coefficient = coefficients
    x_norm = 2.0 * np.arange(shape[1], dtype=np.float32) / max(1.0, shape[1] - 1.0) - 1.0
    y_norm = 2.0 * np.arange(shape[0], dtype=np.float32) / max(1.0, shape[0] - 1.0) - 1.0
    return np.asarray(
        intercept + x_coefficient * x_norm[None, :] + y_coefficient * y_norm[:, None],
        dtype=np.float32,
    )


def _rms_provider(image, coverage_mask, source_mask, fallback_rms):
    try:
        return _background2d(
            image,
            coverage_mask,
            box_size=_mesh_box_size(np.shape(image), _AUTO_MESH_COUNTS[0]),
            filter_size=(3, 3),
            source_mask=source_mask,
        )
    except Exception:
        return _ScalarRms(np.shape(image), fallback_rms)


def _make_background_model(image_data, coverage_mask, target, config):
    if np.all(coverage_mask):
        raise ValueError(f"Cannot estimate background for {target}: no finite nonzero pixels")

    if config.mode == "legacy":
        box_size, filter_size = _legacy_background_parameters(np.shape(image_data))
        try:
            model = _background2d(
                image_data,
                coverage_mask,
                box_size=box_size,
                filter_size=filter_size,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not construct a legacy background model for {target} "
                f"with shape {np.shape(image_data)}: {exc}"
            ) from exc
        diagnostics = {
            "algorithm": "legacy",
            "box_size": tuple(int(v) for v in box_size),
            "filter_size": tuple(int(v) for v in filter_size),
            "sky_fraction": float(np.count_nonzero(~coverage_mask) / coverage_mask.size),
            "residual_median": np.nan,
            "residual_rms": float(model.background_rms_median),
        }
        return _BackgroundEstimate(model, diagnostics=diagnostics)

    thumbnail, stride = _thumbnail(image_data)
    thumb_coverage = _coverage_mask(thumbnail)
    initial_median, initial_rms = _robust_location_scale(thumbnail[~thumb_coverage])
    source_mask = _source_mask(thumbnail, thumb_coverage, initial_median, initial_rms)
    sky_valid = ~thumb_coverage & ~source_mask
    _, noise_rms = _robust_location_scale((thumbnail - initial_median)[sky_valid])
    if not np.isfinite(noise_rms) or noise_rms <= 0:
        noise_rms = initial_rms

    selected, candidates = _automatic_thumbnail_model(
        thumbnail,
        thumb_coverage,
        source_mask,
        noise_rms,
        config.mode,
    )
    if selected.get("fallback"):
        warning_text = (
            "no unmasked sky remains; using the median of all finite coverage"
            if selected.get("fallback_reason") == "no unmasked sky"
            else "usable sky is below the required fraction; using a constant background"
        )
        warnings.warn(
            f"{target}: {warning_text} "
            f"(sky fraction={selected['sky_fraction']:.3f}).",
            RuntimeWarning,
        )

    full_source_mask = _expanded_mask(source_mask, stride, np.shape(image_data))
    full_sky_valid = ~coverage_mask & ~full_source_mask
    full_sky_count = np.count_nonzero(full_sky_valid)
    full_coverage_count = max(1, np.count_nonzero(~coverage_mask))
    full_sky_fraction = full_sky_count / full_coverage_count
    fit_valid = full_sky_valid if full_sky_count else ~coverage_mask

    algorithm = selected["algorithm"]
    if algorithm != "constant" and full_sky_fraction < _MIN_SKY_FRACTION:
        selected = dict(candidates[0])
        selected["fallback"] = True
        selected["fallback_reason"] = "usable full-resolution sky below 30%"
        algorithm = "constant"
        warnings.warn(
            f"{target}: full-resolution usable sky fraction "
            f"{full_sky_fraction:.3f} is below {_MIN_SKY_FRACTION:.2f}; "
            "using a constant background.",
            RuntimeWarning,
        )

    if algorithm == "mesh":
        mesh_count = selected["mesh_count"]
        full_box = _mesh_box_size(np.shape(image_data), mesh_count)
        try:
            full_model = _background2d(
                image_data,
                coverage_mask,
                box_size=full_box,
                filter_size=selected["filter_size"],
                source_mask=full_source_mask,
            )
        except Exception as exc:
            selected = dict(candidates[0])
            selected["fallback"] = True
            selected["fallback_reason"] = f"full-resolution mesh failed: {exc}"
            algorithm = "constant"
            warnings.warn(
                f"{target}: selected mesh failed at full resolution; "
                "using a constant background.",
                RuntimeWarning,
            )
        else:
            background_source = full_model
            rms_source = full_model
            full_box_size = full_box
    if algorithm == "plane":
        try:
            background_source, coefficients = _fit_plane(image_data, fit_valid)
        except ValueError as exc:
            selected = dict(candidates[0])
            selected["fallback"] = True
            selected["fallback_reason"] = f"full-resolution plane failed: {exc}"
            algorithm = "constant"
            warnings.warn(
                f"{target}: selected plane failed at full resolution; "
                "using a constant background.",
                RuntimeWarning,
            )
        else:
            selected = dict(selected)
            selected["coefficients"] = coefficients
            selected["model_uncertainty"] = float(
                noise_rms * np.sqrt(3.0 / full_sky_count)
                if full_sky_count and np.isfinite(noise_rms)
                else np.nan
            )
            rms_source = _rms_provider(
                image_data, coverage_mask, full_source_mask, noise_rms
            )
            full_box_size = None
    if algorithm == "constant":
        value, full_noise_rms = _robust_location_scale(np.asarray(image_data)[fit_valid])
        selected = dict(selected)
        selected["value"] = float(value)
        selected["coefficients"] = (float(value), 0.0, 0.0)
        selected["model_uncertainty"] = float(
            full_noise_rms / np.sqrt(np.count_nonzero(fit_valid))
            if np.count_nonzero(fit_valid) and np.isfinite(full_noise_rms)
            else np.nan
        )
        background_source = np.full(np.shape(image_data), value, dtype=np.float32)
        rms_source = _rms_provider(
            image_data, coverage_mask, full_source_mask, noise_rms
        )
        full_box_size = None

    selected = dict(selected)
    selected["sky_fraction"] = float(full_sky_fraction)

    diagnostics = {
        key: value
        for key, value in selected.items()
        if key != "background"
    }
    diagnostics["box_size"] = (
        None if full_box_size is None else tuple(int(v) for v in full_box_size)
    )
    diagnostics["thumbnail_stride"] = int(stride)
    diagnostics["candidate_summary"] = [
        {
            "algorithm": item["algorithm"],
            "box_size": item.get("box_size"),
            "filter_size": item.get("filter_size"),
            "score": item["score"],
            "passes": item["passes"],
            "rejection_reason": item["rejection_reason"],
        }
        for item in candidates
    ]
    emit_detail(
        f"Background selection {target}: algorithm={algorithm}, "
        f"box={diagnostics['box_size']}, filter={diagnostics.get('filter_size')}, "
        f"sky={diagnostics['sky_fraction']:.3f}, score={diagnostics['score']:.3f}"
    )
    for item in diagnostics["candidate_summary"]:
        status = "accepted" if item["passes"] else item["rejection_reason"]
        emit_detail(
            f"  candidate={item['algorithm']} box={item['box_size']} "
            f"filter={item['filter_size']}: {status}"
        )
    return _BackgroundEstimate(background_source, rms_source, diagnostics)


def _global_rms_value(image_data, coverage_mask, target):
    finite_sky = np.asarray(image_data)[~coverage_mask]
    if finite_sky.size == 0:
        raise ValueError(f"Cannot estimate global background RMS for {target}: no sky pixels")
    _, _, rms = sigma_clipped_stats(finite_sky, sigma=3.0)
    if not np.isfinite(rms) or rms <= 0:
        rms = np.nanstd(finite_sky)
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError(f"Cannot estimate a finite positive background RMS for {target}")
    return float(rms)


def _model_rms_value(background, image_data, coverage_mask, target):
    try:
        rms = float(background.background_rms_median)
    except Exception:
        rms = np.nan
    if np.isfinite(rms) and rms > 0:
        return rms
    return _global_rms_value(image_data, coverage_mask, target)


def _estimate_error_map(background, image_data, coverage_mask, target, config):
    if config.error_mode == "global_rms":
        rms_value = (
            _global_rms_value(image_data, coverage_mask, target)
            if background is None
            else _model_rms_value(background, image_data, coverage_mask, target)
        )
        error = np.full(np.shape(image_data), rms_value, dtype=np.float32)
        error[coverage_mask] = 0.0
        return error, "global_rms"

    try:
        if background is None:
            background = _make_background_model(image_data, coverage_mask, target, config)
        error = np.asarray(background.background_rms, dtype=np.float32)
        valid = np.isfinite(error) & (error > 0) & ~coverage_mask
        if not np.any(valid):
            raise ValueError("local background RMS contains no finite positive values")
        fallback_value = float(np.nanmedian(error[valid]))
        error[~valid & ~coverage_mask] = fallback_value
        error[coverage_mask] = 0.0
        return error, "local_rms"
    except Exception as exc:
        if config.fallback == "error":
            raise ValueError(f"Could not estimate local background RMS for {target}: {exc}") from exc
        rms_value = (
            _global_rms_value(image_data, coverage_mask, target)
            if background is None
            else _model_rms_value(background, image_data, coverage_mask, target)
        )
        error = np.full(np.shape(image_data), rms_value, dtype=np.float32)
        error[coverage_mask] = 0.0
        return error, "global_rms"


def _update_error(image_set, error, galaxy_name, observatory, band, source):
    if callable(getattr(image_set, "get_error_source", None)):
        image_set.update_error(
            error,
            galaxy_name,
            observatory,
            band,
            source=source,
        )
    else:
        image_set.update_error(error, galaxy_name, observatory, band)


def _update_background_header(
    image_set,
    header,
    galaxy_name,
    observatory,
    band,
    diagnostics,
    *,
    residual=False,
):
    updater = getattr(image_set, "update_header", None)
    if not callable(updater) or header is None:
        return
    updated = header.copy()
    prefix = "RB" if residual else "BK"
    algorithm = str(diagnostics.get("algorithm", "unknown")).upper()[:8]
    updated[f"{prefix}GALG"] = (algorithm, "Spec7DT background algorithm")
    updated[f"{prefix}GSKY"] = (
        float(diagnostics.get("sky_fraction", np.nan)),
        "Usable sky fraction",
    )
    updated[f"{prefix}GMED"] = (
        float(diagnostics.get("residual_median", np.nan)),
        "Residual sky median",
    )
    updated[f"{prefix}GRMS"] = (
        float(diagnostics.get("residual_rms", np.nan)),
        "Residual sky RMS",
    )
    uncertainty = diagnostics.get("model_uncertainty")
    if uncertainty is not None and np.isfinite(uncertainty):
        updated[f"{prefix}GUNC"] = (
            float(uncertainty),
            "Background model uncertainty",
        )
    box_size = diagnostics.get("box_size")
    if box_size is not None:
        updated[f"{prefix}GBOX"] = (
            "x".join(str(int(v)) for v in box_size),
            "Background box size yx",
        )
    filter_size = diagnostics.get("filter_size")
    if filter_size is not None:
        updated[f"{prefix}GFIL"] = (
            "x".join(str(int(v)) for v in filter_size),
            "Background mesh filter yx",
        )
    coefficients = diagnostics.get("coefficients")
    if coefficients is not None:
        for index, coefficient in enumerate(coefficients[:3]):
            if np.isfinite(coefficient):
                updated[f"{prefix}GC{index}"] = (
                    float(coefficient),
                    "Background plane coefficient",
                )
    updated.add_history(
        f"Spec7DT {('residual ' if residual else '')}background: {algorithm}"
    )
    updater(updated, galaxy_name, observatory, band)


def estimateMissingError(
    image_set,
    image_data,
    error_data,
    galaxy_name,
    observatory,
    band,
    background_config=None,
):
    """Estimate a missing 1-sigma error map without subtracting the background."""
    if _error_source(image_set, error_data, galaxy_name, observatory, band) != "missing":
        return

    config = BackgroundConfig.from_value(background_config)
    target = _target_path(image_set, galaxy_name, observatory, band)
    coverage_mask = _coverage_mask(image_data)
    model_start = perf_counter()
    background = None
    model_error = None
    if config.error_mode == "local_rms":
        try:
            background = _make_background_model(image_data, coverage_mask, target, config)
        except ValueError as exc:
            if config.fallback == "error":
                raise
            model_error = exc
    model_elapsed = perf_counter() - model_start

    rms_start = perf_counter()
    if model_error is None:
        error, source = _estimate_error_map(
            background,
            image_data,
            coverage_mask,
            target,
            config,
        )
    else:
        rms_value = _global_rms_value(image_data, coverage_mask, target)
        error = np.full(np.shape(image_data), rms_value, dtype=np.float32)
        error[coverage_mask] = 0.0
        source = "global_rms"
    _update_error(image_set, error, galaxy_name, observatory, band, source)
    rms_elapsed = perf_counter() - rms_start
    emit_detail(
        f"Background error {galaxy_name}/{observatory}/{band}: "
        f"model={model_elapsed:.3f}s, rms={rms_elapsed:.3f}s, source={source}"
    )


def backgroundSubtraction(
    image_set,
    image_data,
    galaxy_name,
    observatory,
    band,
    header=None,
    error_data=None,
    background_config=None,
):
    """Subtract an automatically selected source-masked background model."""
    config = BackgroundConfig.from_value(background_config)
    target = _target_path(image_set, galaxy_name, observatory, band)
    coverage_mask = _coverage_mask(image_data)

    model_start = perf_counter()
    background = _make_background_model(image_data, coverage_mask, target, config)
    model_elapsed = perf_counter() - model_start

    subtraction_start = perf_counter()
    background_map = np.asarray(background.background)
    image_array = np.asarray(image_data)
    if np.issubdtype(image_array.dtype, np.floating) and image_array.flags.writeable:
        original_coverage = image_array[coverage_mask].copy()
        np.subtract(image_array, background_map, out=image_array)
        image_array[coverage_mask] = original_coverage
        subtracted = image_array
    else:
        subtracted = np.asarray(image_array, dtype=np.float32) - background_map
        subtracted[coverage_mask] = image_array[coverage_mask]
    del background_map
    image_set.update_data(subtracted, galaxy_name, observatory, band)
    subtraction_elapsed = perf_counter() - subtraction_start

    rms_elapsed = 0.0
    source = _error_source(image_set, error_data, galaxy_name, observatory, band)
    if source == "missing":
        rms_start = perf_counter()
        estimated_error, source = _estimate_error_map(
            background,
            image_data,
            coverage_mask,
            target,
            config,
        )
        _update_error(
            image_set,
            estimated_error,
            galaxy_name,
            observatory,
            band,
            source,
        )
        rms_elapsed = perf_counter() - rms_start

    diagnostics = dict(background.diagnostics)
    if config.mode != "legacy":
        _update_background_header(
            image_set,
            header,
            galaxy_name,
            observatory,
            band,
            diagnostics,
        )
    emit_detail(
        f"Background {galaxy_name}/{observatory}/{band}: "
        f"model={model_elapsed:.3f}s, subtraction={subtraction_elapsed:.3f}s, "
        f"rms={rms_elapsed:.3f}s, error_source={source}"
    )
    return AnalysisProduct(kind="background", data=diagnostics)


def _ellipse_mask(shape, geometry, scale=1.0):
    x0, y0, a, b, theta = (float(value) for value in geometry)
    a = max(np.finfo(float).eps, abs(a) * scale)
    b = max(np.finfo(float).eps, abs(b) * scale)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = np.arange(shape[1], dtype=float) - x0
    mask = np.zeros(shape, dtype=bool)
    for y_start in range(0, shape[0], 512):
        y_stop = min(shape[0], y_start + 512)
        y = np.arange(y_start, y_stop, dtype=float)[:, None] - y0
        x_rot = x[None, :] * cos_theta + y * sin_theta
        y_rot = -x[None, :] * sin_theta + y * cos_theta
        mask[y_start:y_stop] = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
    return mask


def residualBackgroundSubtraction(image_set, background_config=None):
    """Remove a constant or plane residual outside the common target aperture."""
    config = BackgroundConfig.from_value(background_config)
    if config.mode == "legacy" or not config.post_residual:
        return AnalysisProduct(kind="residual_background", data={})

    all_diagnostics = {}
    for galaxy, observatories in image_set.data.items():
        shapes = {
            tuple(np.shape(image))
            for bands in observatories.values()
            for image in bands.values()
        }
        if len(shapes) != 1:
            warnings.warn(
                f"{galaxy}: images do not share one registered grid shape; "
                "residual background was skipped.",
                RuntimeWarning,
            )
            continue
        geometries = useful_functions.extract_values_recursive(
            image_set.cutout_shape,
            galaxy,
        )
        valid_geometries = []
        for geometry in geometries:
            try:
                values = np.asarray(geometry, dtype=float)
            except (TypeError, ValueError):
                continue
            if values.shape == (5,) and np.all(np.isfinite(values)):
                valid_geometries.append(values)
        if not valid_geometries:
            warnings.warn(
                f"{galaxy}: no valid cutout geometry; residual background was skipped.",
                RuntimeWarning,
            )
            continue
        common_geometry = np.nanmedian(np.vstack(valid_geometries), axis=0)

        for observatory, bands in observatories.items():
            for band, image_data in bands.items():
                target = _target_name(galaxy, observatory, band)
                image = np.asarray(image_data)
                coverage_mask = _coverage_mask(image)
                target_mask = _ellipse_mask(
                    image.shape,
                    common_geometry,
                    scale=_RESIDUAL_TARGET_SCALE,
                )
                sky_valid = ~coverage_mask & ~target_mask
                coverage_count = max(1, np.count_nonzero(~coverage_mask))
                sky_fraction = np.count_nonzero(sky_valid) / coverage_count
                if not np.any(sky_valid):
                    sky_valid = ~coverage_mask
                    warnings.warn(
                        f"{target}: no sky remains outside the target aperture; "
                        "using the global finite median.",
                        RuntimeWarning,
                    )
                elif sky_fraction < _MIN_SKY_FRACTION:
                    warnings.warn(
                        f"{target}: residual sky fraction {sky_fraction:.3f} is below "
                        f"{_MIN_SKY_FRACTION:.2f}; using a constant residual.",
                        RuntimeWarning,
                    )

                value, noise_rms = _robust_location_scale(image[sky_valid])
                constant = np.full(image.shape, value, dtype=np.float32)
                constant_metrics = _validation_metrics(
                    np.asarray(image, dtype=float) - constant,
                    sky_valid,
                    noise_rms,
                )
                selected_background = constant
                diagnostics = {
                    "algorithm": "constant",
                    "value": float(value),
                    "coefficients": (float(value), 0.0, 0.0),
                    "sky_fraction": float(sky_fraction),
                    "box_size": None,
                    "filter_size": None,
                    "model_uncertainty": float(
                        noise_rms / np.sqrt(np.count_nonzero(sky_valid))
                        if np.count_nonzero(sky_valid) and np.isfinite(noise_rms)
                        else np.nan
                    ),
                    **constant_metrics,
                }

                if config.mode == "auto" and sky_fraction >= _MIN_SKY_FRACTION:
                    try:
                        plane, coefficients = _fit_plane(image, sky_valid)
                        plane_metrics = _validation_metrics(
                            np.asarray(image, dtype=float) - plane,
                            sky_valid,
                            noise_rms,
                        )
                        if (
                            np.isfinite(plane_metrics["cell_rms"])
                            and np.isfinite(constant_metrics["cell_rms"])
                            and plane_metrics["cell_rms"]
                            <= 0.75 * constant_metrics["cell_rms"]
                        ):
                            selected_background = plane
                            diagnostics.update(
                                algorithm="plane",
                                coefficients=coefficients,
                                model_uncertainty=float(
                                    noise_rms
                                    * np.sqrt(3.0 / np.count_nonzero(sky_valid))
                                    if np.count_nonzero(sky_valid)
                                    and np.isfinite(noise_rms)
                                    else np.nan
                                ),
                                **plane_metrics,
                            )
                    except ValueError:
                        pass

                corrected = np.asarray(image, dtype=np.float32) - selected_background
                corrected[coverage_mask] = image[coverage_mask]
                image_set.update_data(corrected, galaxy, observatory, band)
                header = image_set.header[galaxy][observatory][band]
                _update_background_header(
                    image_set,
                    header,
                    galaxy,
                    observatory,
                    band,
                    diagnostics,
                    residual=True,
                )
                all_diagnostics[target] = diagnostics
                emit_detail(
                    f"Residual background {target}: algorithm={diagnostics['algorithm']}, "
                    f"sky={sky_fraction:.3f}, median={diagnostics['residual_median']:.6g}"
                )

    return AnalysisProduct(kind="residual_background", data=all_diagnostics)
