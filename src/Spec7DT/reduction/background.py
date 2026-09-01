from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground

from ..utils.reporting import emit_detail


@dataclass(frozen=True)
class BackgroundConfig:
    """Background subtraction and missing-error estimation settings."""

    error_mode: Literal["local_rms", "global_rms"] = "local_rms"
    fallback: Literal["global_rms", "error"] = "global_rms"

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


def _background_parameters(shape):
    ny, nx = shape
    box_size = (max(1, int(ny * 0.25)), max(1, int(nx * 0.25)))
    filter_size = (
        max(1, int(ny * 5e-3) * 2 + 1),
        max(1, int(nx * 5e-3) * 2 + 1),
    )
    return box_size, filter_size


def _make_background_model(image_data, coverage_mask, target):
    if np.all(coverage_mask):
        raise ValueError(f"Cannot estimate background for {target}: no finite nonzero pixels")

    box_size, filter_size = _background_parameters(np.shape(image_data))
    try:
        return Background2D(
            image_data,
            box_size=box_size,
            filter_size=filter_size,
            coverage_mask=coverage_mask,
            fill_value=0.0,
            sigma_clip=SigmaClip(sigma=3.0),
            bkg_estimator=MedianBackground(),
            exclude_percentile=50.0,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not construct a local background model for {target} "
            f"with shape {np.shape(image_data)}: {exc}"
        ) from exc


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
        if background is None:
            rms_value = _global_rms_value(image_data, coverage_mask, target)
        else:
            rms_value = _model_rms_value(background, image_data, coverage_mask, target)
        error = np.full(np.shape(image_data), rms_value, dtype=np.float32)
        error[coverage_mask] = 0.0
        return error, "global_rms"

    try:
        if background is None:
            background = _make_background_model(image_data, coverage_mask, target)
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
        if background is None:
            rms_value = _global_rms_value(image_data, coverage_mask, target)
        else:
            rms_value = _model_rms_value(background, image_data, coverage_mask, target)
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
            background = _make_background_model(image_data, coverage_mask, target)
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
    error_data=None,
    background_config=None,
):
    """Subtract the background and reuse its model for any missing error map."""
    config = BackgroundConfig.from_value(background_config)
    target = _target_path(image_set, galaxy_name, observatory, band)
    coverage_mask = _coverage_mask(image_data)

    model_start = perf_counter()
    background = _make_background_model(image_data, coverage_mask, target)
    model_elapsed = perf_counter() - model_start

    subtraction_start = perf_counter()
    background_map = np.asarray(background.background)
    image_array = np.asarray(image_data)
    if np.issubdtype(image_array.dtype, np.floating) and image_array.flags.writeable:
        np.subtract(image_array, background_map, out=image_array)
        subtracted = image_array
    else:
        subtracted = np.asarray(image_array, dtype=np.float32) - background_map
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

    emit_detail(
        f"Background {galaxy_name}/{observatory}/{band}: "
        f"model={model_elapsed:.3f}s, subtraction={subtraction_elapsed:.3f}s, "
        f"rms={rms_elapsed:.3f}s, error_source={source}"
    )
