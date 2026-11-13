from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, shift

from .models import EffectivePSF, PixelSampling, PSFEnsemble, PSFSample


def centered_coordinates(shape):
    y, x = np.indices(shape, dtype=float)
    return y - (shape[0] - 1) / 2.0, x - (shape[1] - 1) / 2.0


def psf_statistics(data, pixel_scale):
    data = np.asarray(data, dtype=float)
    total = float(np.sum(data, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF has no positive finite flux")
    normalized = data / total
    y, x = centered_coordinates(normalized.shape)
    radius = np.hypot(x, y)
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(normalized.ravel()[order])
    radii = radius.ravel()[order]
    r50 = float(radii[min(np.searchsorted(cumulative, 0.5), len(radii) - 1)]) * pixel_scale
    r80 = float(radii[min(np.searchsorted(cumulative, 0.8), len(radii) - 1)]) * pixel_scale
    peak = float(np.max(normalized))
    area = int(np.count_nonzero(normalized >= 0.5 * peak))
    fwhm = 2.0 * np.sqrt(max(area, 1) / np.pi) * pixel_scale
    return fwhm, r50, r80


def validate_effective_psf(
    data,
    sampling,
    source,
    *,
    edge_tolerance=1e-4,
    negative_peak_tolerance=1e-8,
    negative_mass_tolerance=1e-6,
    centroid_limit=0.5,
    frame="image",
    provenance=None,
):
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or min(array.shape) < 3:
        raise ValueError(f"{source}: PSF must be a non-empty 2-D image")
    if any(size % 2 == 0 for size in array.shape):
        raise ValueError(f"{source}: PSF dimensions must be odd")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{source}: PSF contains non-finite values")
    sampling = PixelSampling.from_value(sampling)
    scale = sampling.equivalent_scale_arcsec
    peak = float(np.max(array))
    if peak <= 0:
        raise ValueError(f"{source}: PSF peak is not positive")
    negative = np.minimum(array, 0.0)
    removed_mass = float(-np.sum(negative, dtype=np.float64))
    positive_mass = float(np.sum(np.maximum(array, 0.0), dtype=np.float64))
    negative_fraction = removed_mass / positive_mass if positive_mass > 0 else np.inf
    if float(np.min(array)) < -float(negative_peak_tolerance) * peak:
        raise ValueError(f"{source}: PSF contains significant negative samples")
    if negative_fraction > float(negative_mass_tolerance):
        raise ValueError(
            f"{source}: negative PSF mass fraction {negative_fraction:.3g} exceeds "
            f"{negative_mass_tolerance:.3g}"
        )
    array = np.maximum(array, 0.0)
    total = float(np.sum(array, dtype=np.float64))
    if total <= 0:
        raise ValueError(f"{source}: PSF integral is not positive")
    array /= total

    y, x = centered_coordinates(array.shape)
    cy = float(np.sum(y * array))
    cx = float(np.sum(x * array))
    if np.hypot(cx, cy) > float(centroid_limit):
        raise ValueError(
            f"{source}: PSF centroid is more than {float(centroid_limit):.3g} pixel from the stamp center"
        )
    if np.hypot(cx, cy) > 1e-5:
        array = shift(array, shift=(-cy, -cx), order=1, mode="constant", cval=0.0)
        array = np.maximum(array, 0.0)
        array /= np.sum(array, dtype=np.float64)

    border = np.zeros(array.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    edge_mass = float(np.sum(array[border]))
    if edge_mass > edge_tolerance:
        raise ValueError(
            f"{source}: PSF stamp is truncated (edge mass {edge_mass:.3g} > {edge_tolerance:.3g})"
        )
    fwhm, r50, r80 = psf_statistics(array, scale)
    return EffectivePSF(
        np.asarray(array, dtype=np.float32),
        scale,
        fwhm,
        r50,
        r80,
        str(source),
        frame=frame,
        quality={
            "edge_mass": edge_mass,
            "centroid_x": cx,
            "centroid_y": cy,
            "negative_mass_removed": removed_mass,
            "negative_mass_fraction": negative_fraction,
        },
        sampling=sampling,
        provenance=dict(provenance or {}),
    )


def render_oversampled_psf(
    data,
    oversampling,
    phase,
    output_sampling,
    *,
    stamp_size=11,
    integration_subsamples=None,
    transform=None,
):
    """Integrate a positive oversampled ePSF onto an output pixel grid."""

    source = np.asarray(data, dtype=float)
    if source.ndim != 2 or not np.all(np.isfinite(source)):
        raise ValueError("Oversampled PSF must be a finite 2-D array")
    if any(size % 2 == 0 for size in source.shape):
        raise ValueError("Oversampled PSF dimensions must be odd")
    oy, ox = (int(oversampling[0]), int(oversampling[1]))
    if oy < 1 or ox < 1:
        raise ValueError("PSF oversampling must be positive")
    phase_y, phase_x = (float(phase[0]), float(phase[1]))
    output_sampling = PixelSampling.from_value(output_sampling)
    subsamples = integration_subsamples or min(8, max(oy, ox))
    subsamples = max(1, int(subsamples))
    size = int(stamp_size)
    if size < 3 or size % 2 == 0:
        raise ValueError("Rendered PSF stamp_size must be an odd integer >= 3")

    yy, xx = centered_coordinates((size, size))
    offsets = (np.arange(subsamples, dtype=float) + 0.5) / subsamples - 0.5
    transform = np.eye(2, dtype=float) if transform is None else np.asarray(transform, dtype=float)
    if transform.shape != (2, 2) or not np.all(np.isfinite(transform)):
        raise ValueError("PSF render transform must be a finite 2x2 matrix")
    center_y = (source.shape[0] - 1) / 2.0
    center_x = (source.shape[1] - 1) / 2.0
    rendered = np.zeros((size, size), dtype=float)
    for dy in offsets:
        for dx in offsets:
            output = np.stack((yy + dy - phase_y, xx + dx - phase_x), axis=0)
            native = np.einsum("ij,j...->i...", transform, output)
            coords = np.stack((center_y + oy * native[0], center_x + ox * native[1]))
            rendered += map_coordinates(
                source,
                coords,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
    rendered /= subsamples ** 2
    rendered = np.maximum(rendered, 0.0)
    total = float(np.sum(rendered, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Oversampled PSF rendering produced no positive flux")
    return rendered / total


def common_shape(arrays):
    arrays = [np.asarray(item, dtype=float) for item in arrays]
    ny = max(item.shape[0] for item in arrays)
    nx = max(item.shape[1] for item in arrays)
    ny += 1 - ny % 2
    nx += 1 - nx % 2
    result = []
    for item in arrays:
        canvas = np.zeros((ny, nx), dtype=float)
        y0 = (ny - item.shape[0]) // 2
        x0 = (nx - item.shape[1]) // 2
        canvas[y0:y0 + item.shape[0], x0:x0 + item.shape[1]] = item
        result.append(canvas)
    return result


def encircled_energy_difference(first, second):
    first, second = common_shape((first, second))
    radius = np.hypot(*centered_coordinates(first.shape))
    order = np.argsort(radius.ravel())
    return float(
        np.max(
            np.abs(
                np.cumsum(first.ravel()[order]) - np.cumsum(second.ravel()[order])
            )
        )
    )


def _require_common_sampling(samples):
    first = samples[0].model.sampling
    first_matrix = np.asarray(first.matrix_arcsec, dtype=float)
    for sample in samples[1:]:
        sampling = sample.model.sampling
        if sampling.frame != first.frame or not np.allclose(
            np.asarray(sampling.matrix_arcsec, dtype=float),
            first_matrix,
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError(
                "PSF ensemble samples must use one common angular sampling and frame"
            )


def summarize_spatial_ensemble(ensemble, *, r80_tolerance, ee_tolerance, l1_tolerance, centroid_tolerance):
    _require_common_sampling(ensemble.samples)
    groups = {}
    for sample in ensemble.samples:
        groups.setdefault(sample.position, []).append(sample)
    position_models = []
    for position, samples in groups.items():
        arrays = common_shape(sample.model.data for sample in samples)
        weights = np.asarray([sample.weight for sample in samples], dtype=float)
        weights /= np.sum(weights)
        combined = sum(weight * array for weight, array in zip(weights, arrays))
        combined /= np.sum(combined)
        scale = samples[0].model.pixel_scale_arcsec
        fwhm, r50, r80 = psf_statistics(combined, scale)
        y, x = centered_coordinates(combined.shape)
        position_models.append((
            position,
            combined,
            r80,
            float(np.sum(x * combined)),
            float(np.sum(y * combined)),
            fwhm,
            r50,
        ))
    if len(position_models) == 1:
        metrics = {"spatial_r80_spread": 0.0, "spatial_ee_max": 0.0, "spatial_l1_max": 0.0, "spatial_centroid_max": 0.0}
        return metrics
    r80s = np.asarray([item[2] for item in position_models], dtype=float)
    r80_spread = float((np.max(r80s) - np.min(r80s)) / max(np.mean(r80s), np.finfo(float).eps))
    ee_max = l1_max = centroid_max = 0.0
    for index, first in enumerate(position_models):
        for second in position_models[index + 1:]:
            a, b = common_shape((first[1], second[1]))
            a /= np.sum(a)
            b /= np.sum(b)
            ee_max = max(ee_max, encircled_energy_difference(a, b))
            l1_max = max(l1_max, float(np.sum(np.abs(a - b))))
            centroid_max = max(centroid_max, float(np.hypot(first[3] - second[3], first[4] - second[4])))
    metrics = {
        "spatial_r80_spread": r80_spread,
        "spatial_ee_max": ee_max,
        "spatial_l1_max": l1_max,
        "spatial_centroid_max": centroid_max,
    }
    failures = []
    if r80_spread > r80_tolerance:
        failures.append(f"r80 spread {r80_spread:.4g} > {r80_tolerance:.4g}")
    if ee_max > ee_tolerance:
        failures.append(f"EE difference {ee_max:.4g} > {ee_tolerance:.4g}")
    if l1_max > l1_tolerance:
        failures.append(f"L1 difference {l1_max:.4g} > {l1_tolerance:.4g}")
    if centroid_max > centroid_tolerance:
        failures.append(f"centroid spread {centroid_max:.4g} > {centroid_tolerance:.4g}")
    if failures:
        raise ValueError(
            "PSF is not stationary over the analysis tile; reduce the tile or provide "
            "a spatially varying PSF workflow: " + "; ".join(failures)
        )
    return metrics


def representative_from_ensemble(ensemble, source, *, validator, edge_tolerance):
    _require_common_sampling(ensemble.samples)
    arrays = common_shape(sample.model.data for sample in ensemble.samples)
    weights = np.asarray([sample.weight for sample in ensemble.samples], dtype=float)
    weights /= np.sum(weights)
    combined = sum(weight * array for weight, array in zip(weights, arrays))
    combined = np.maximum(combined, 0.0)
    combined /= np.sum(combined)
    first = ensemble.samples[0].model
    result = validator(
        combined,
        first.sampling,
        source,
        edge_tolerance=edge_tolerance,
        frame=first.frame,
        provenance=ensemble.provenance,
    )
    result.quality.update(ensemble.spatial_metrics)
    result.quality["ensemble_size"] = len(ensemble.samples)
    return result


def singleton_ensemble(model):
    return PSFEnsemble([PSFSample(model=model)])

