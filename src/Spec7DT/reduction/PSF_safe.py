"""Positive-kernel PSF matching facade for spatially resolved SED analysis.

Product recognition and instrument-specific ePSF rules live in registered PSF
providers.  This module contains only instrument-independent matching and keeps
the public safe-PSF API beside the untouched legacy implementation in
``PSF.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.nddata.utils import NoOverlapError
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import affine_transform, gaussian_filter, laplace, shift
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.optimize import least_squares, lsq_linear
from scipy.sparse.linalg import LinearOperator

from photutils.detection import DAOStarFinder

from ..core import AnalysisProduct
from ..psf.models import (
    EffectivePSF,
    ImageProductMetadata,
    PixelSampling,
    PSFEnsemble,
    PSFRequest,
    PSFSample,
    ResolvedPSF,
)
from ..psf.providers import (
    ExternalPSFProvider,
    FunctionalPSFProvider,
    PSFProviderRegistry,
    ProviderUnavailable,
)
from ..psf.render import (
    centered_coordinates as _centered_coordinates,
    encircled_energy_difference as _encircled_energy_difference,
    psf_statistics as _psf_statistics,
    singleton_ensemble,
    validate_effective_psf,
)
from ..utils.reporting import emit_detail
from ..utils.utility import useful_functions
from .PSF import PSFConvolutionConfig, PSFConvolutionEngine, PointSpreadFunction

if TYPE_CHECKING:
    from ..handlers.spherex_mosaic_psf import SpherexMosaicPSFSynthesisConfig


def _normalize_fov(value):
    if value is None:
        return None
    values = (value, value) if isinstance(value, u.Quantity) and value.isscalar else tuple(value)
    if len(values) != 2:
        raise ValueError("pretrim_fov must contain (ny, nx)")
    result = []
    for item in values:
        if not isinstance(item, u.Quantity) or not item.unit.is_equivalent(u.arcsec):
            raise TypeError("pretrim_fov values must be angular quantities")
        item = item.to(u.arcsec)
        if not item.isscalar or not np.isfinite(item.value) or item.value <= 0:
            raise ValueError("pretrim_fov values must be finite and positive")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class SafePSFConfig:
    """Configuration for full-profile, non-negative PSF matching."""

    psf_manifest: str | Path | Mapping[str, Any] | None = None
    calibration_registry: str | Path | Mapping[str, Any] | None = None
    mosaic_synthesis: SpherexMosaicPSFSynthesisConfig | Mapping[str, Any] | None = None
    pretrim_fov: u.Quantity | tuple[u.Quantity, u.Quantity] | None = None
    measurement_region: Literal["pretrim", "full"] = "full"
    min_stars: int = 20
    fit_oversampling: int = 4
    undersampled_fwhm_pixels: float = 2.0
    spatial_mode: Literal["validated_stationary"] = "validated_stationary"
    spatial_sample_pattern: Literal["center_corners"] = "center_corners"
    spatial_r80_tolerance: float = 0.02
    spatial_ee_tolerance: float = 0.01
    spatial_l1_tolerance: float = 0.02
    allow_unvalidated_stationary: bool = False
    phase_sampling: Literal["native", "center"] = "native"
    psf_edge_mass_tolerance: float = 1e-4
    negative_peak_tolerance: float = 1e-8
    negative_mass_tolerance: float = 1e-6
    level2_stamp_size: int = 11
    use_gpu: bool = False
    gpu_fallback: Literal["cpu", "error"] = "cpu"
    kernel_regularization: tuple[float, ...] = (1e-2, 1e-4, 1e-6)
    max_kernel_size: int = 63
    max_target_trials: int = 12
    target_bisection_steps: int = 5
    kernel_sum_tolerance: float = 1e-6
    kernel_centroid_tolerance: float = 0.05
    kernel_edge_tolerance: float = 1e-4
    encircled_energy_tolerance: float = 0.01
    l1_tolerance: float = 0.02
    support_tolerance: float = 1e-6

    def __post_init__(self):
        if self.measurement_region not in {"pretrim", "full"}:
            raise ValueError("measurement_region must be 'pretrim' or 'full'")
        if self.gpu_fallback not in {"cpu", "error"}:
            raise ValueError("gpu_fallback must be 'cpu' or 'error'")
        if self.spatial_mode != "validated_stationary":
            raise ValueError("spatial_mode currently supports only 'validated_stationary'")
        if self.spatial_sample_pattern != "center_corners":
            raise ValueError("spatial_sample_pattern currently supports only 'center_corners'")
        if self.phase_sampling not in {"native", "center"}:
            raise ValueError("phase_sampling must be 'native' or 'center'")
        if not isinstance(self.min_stars, int) or isinstance(self.min_stars, bool) or self.min_stars < 3:
            raise ValueError("min_stars must be an integer >= 3")
        if not isinstance(self.fit_oversampling, int) or self.fit_oversampling < 1:
            raise ValueError("fit_oversampling must be a positive integer")
        if self.undersampled_fwhm_pixels <= 0:
            raise ValueError("undersampled_fwhm_pixels must be positive")
        if self.level2_stamp_size < 3 or self.level2_stamp_size % 2 == 0:
            raise ValueError("level2_stamp_size must be an odd integer >= 3")
        if self.max_kernel_size < 3 or self.max_kernel_size % 2 == 0:
            raise ValueError("max_kernel_size must be an odd integer >= 3")
        if self.max_target_trials < 1:
            raise ValueError("max_target_trials must be positive")
        if self.target_bisection_steps < 0:
            raise ValueError("target_bisection_steps cannot be negative")
        regularization = tuple(float(value) for value in self.kernel_regularization)
        if not regularization or any(not np.isfinite(value) or value < 0 for value in regularization):
            raise ValueError("kernel_regularization must contain finite non-negative values")
        for name in (
            "kernel_sum_tolerance",
            "kernel_centroid_tolerance",
            "kernel_edge_tolerance",
            "encircled_energy_tolerance",
            "l1_tolerance",
            "support_tolerance",
            "spatial_r80_tolerance",
            "spatial_ee_tolerance",
            "spatial_l1_tolerance",
            "psf_edge_mass_tolerance",
            "negative_peak_tolerance",
            "negative_mass_tolerance",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "pretrim_fov", _normalize_fov(self.pretrim_fov))
        object.__setattr__(self, "kernel_regularization", regularization)
        object.__setattr__(self, "use_gpu", bool(self.use_gpu))
        object.__setattr__(self, "allow_unvalidated_stationary", bool(self.allow_unvalidated_stationary))
        if self.mosaic_synthesis is not None:
            from ..handlers.spherex_mosaic_psf import SpherexMosaicPSFSynthesisConfig

            object.__setattr__(
                self,
                "mosaic_synthesis",
                SpherexMosaicPSFSynthesisConfig.from_value(self.mosaic_synthesis),
            )

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("safe_psf_config must be SafePSFConfig, a mapping, or None")

    def convolution_config(self):
        return PSFConvolutionConfig(
            use_gpu=self.use_gpu,
            gpu_fallback=self.gpu_fallback,
        )


@dataclass
class PSFMatchResult:
    input_psf: EffectivePSF
    target_psf: EffectivePSF
    kernel: np.ndarray
    matched_psf: np.ndarray
    metrics: dict[str, float]


class PSFMatchError(RuntimeError):
    """Raised after collecting all bands which failed safe PSF processing."""

    def __init__(self, failures):
        self.failures = dict(failures)
        lines = ["Safe PSF matching failed; no legacy fallback was applied:"]
        lines.extend(f"  {key}: {value}" for key, value in sorted(self.failures.items()))
        super().__init__("\n".join(lines))


def _validate_psf(
    data,
    pixel_scale,
    source,
    *,
    edge_tolerance=1e-4,
    frame="image",
    negative_peak_tolerance=1e-8,
    negative_mass_tolerance=1e-6,
    centroid_limit=0.5,
    provenance=None,
):
    return validate_effective_psf(
        data,
        pixel_scale,
        source,
        edge_tolerance=edge_tolerance,
        frame=frame,
        negative_peak_tolerance=negative_peak_tolerance,
        negative_mass_tolerance=negative_mass_tolerance,
        centroid_limit=centroid_limit,
        provenance=provenance,
    )


def _pad_center(array, shape):
    array = np.asarray(array)
    if array.shape[0] > shape[0] or array.shape[1] > shape[1]:
        y0 = max(0, (array.shape[0] - shape[0]) // 2)
        x0 = max(0, (array.shape[1] - shape[1]) // 2)
        array = array[y0:y0 + shape[0], x0:x0 + shape[1]]
    result = np.zeros(shape, dtype=float)
    y0 = (shape[0] - array.shape[0]) // 2
    x0 = (shape[1] - array.shape[1]) // 2
    result[y0:y0 + array.shape[0], x0:x0 + array.shape[1]] = array
    return result


def _common_shape(first, second):
    ny = max(first.shape[0], second.shape[0])
    nx = max(first.shape[1], second.shape[1])
    ny += 1 - ny % 2
    nx += 1 - nx % 2
    return _pad_center(first, (ny, nx)), _pad_center(second, (ny, nx))


_RESAMPLED_PSF_CACHE = {}


def _resample_psf_affine(data, old_sampling, new_sampling):
    array = np.asarray(data, dtype=float)
    old_sampling = PixelSampling.from_value(old_sampling)
    new_sampling = PixelSampling.from_value(new_sampling)
    old_matrix = np.asarray(old_sampling.matrix_arcsec, dtype=float)
    new_matrix = np.asarray(new_sampling.matrix_arcsec, dtype=float)
    digest = sha256()
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    digest.update(np.ascontiguousarray(old_matrix).view(np.uint8))
    digest.update(np.ascontiguousarray(new_matrix).view(np.uint8))
    key = digest.hexdigest()
    cached = _RESAMPLED_PSF_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    if np.allclose(old_matrix, new_matrix, rtol=1e-7, atol=1e-10):
        result = array.copy()
    else:
        output_from_input_xy = np.linalg.solve(new_matrix, old_matrix)
        input_from_output_xy = np.linalg.solve(old_matrix, new_matrix)
        input_bounds_xy = np.array([
            [-array.shape[1] / 2.0, -array.shape[0] / 2.0],
            [-array.shape[1] / 2.0, array.shape[0] / 2.0],
            [array.shape[1] / 2.0, -array.shape[0] / 2.0],
            [array.shape[1] / 2.0, array.shape[0] / 2.0],
        ]).T
        output_bounds_xy = output_from_input_xy @ input_bounds_xy
        output_nx = max(3, int(np.ceil(2.0 * np.max(np.abs(output_bounds_xy[0])))))
        output_ny = max(3, int(np.ceil(2.0 * np.max(np.abs(output_bounds_xy[1])))))
        output_nx += 1 - output_nx % 2
        output_ny += 1 - output_ny % 2
        swap = np.array(((0.0, 1.0), (1.0, 0.0)))
        matrix_yx = swap @ input_from_output_xy @ swap
        input_center = (np.asarray(array.shape, dtype=float) - 1.0) / 2.0
        output_center = (np.asarray((output_ny, output_nx), dtype=float) - 1.0) / 2.0
        result = affine_transform(
            array,
            matrix_yx,
            offset=input_center - matrix_yx @ output_center,
            output_shape=(output_ny, output_nx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    if result.shape[0] % 2 == 0:
        result = np.pad(result, ((0, 1), (0, 0)))
    if result.shape[1] % 2 == 0:
        result = np.pad(result, ((0, 0), (0, 1)))
    result = np.maximum(result, 0.0)
    total = np.sum(result, dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF resampling produced no finite flux")
    result = result / total
    if len(_RESAMPLED_PSF_CACHE) >= 512:
        _RESAMPLED_PSF_CACHE.pop(next(iter(_RESAMPLED_PSF_CACHE)))
    _RESAMPLED_PSF_CACHE[key] = np.asarray(result, dtype=np.float32)
    return _RESAMPLED_PSF_CACHE[key].copy()


def _resample_psf(data, old_scale, new_scale):
    """Scalar compatibility wrapper around affine PSF resampling."""

    return _resample_psf_affine(
        data,
        PixelSampling.isotropic(float(old_scale)),
        PixelSampling.isotropic(float(new_scale)),
    )


def _resample_model(model, output_sampling):
    return _resample_psf_affine(model.data, model.sampling, output_sampling)


def _kernel_metrics(kernel, matched, target):
    kernel = np.asarray(kernel, dtype=float)
    y, x = _centered_coordinates(kernel.shape)
    total = float(np.sum(kernel, dtype=np.float64))
    cx = float(np.sum(x * kernel) / total)
    cy = float(np.sum(y * kernel) / total)
    if kernel.shape == (1, 1):
        edge_mass = 0.0
    else:
        edge = np.zeros(kernel.shape, dtype=bool)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        edge_mass = float(np.sum(kernel[edge]))
    matched, target = _common_shape(matched, target)
    matched /= np.sum(matched, dtype=np.float64)
    target /= np.sum(target, dtype=np.float64)
    return {
        "kernel_min": float(np.min(kernel)),
        "kernel_sum_error": abs(total - 1.0),
        "kernel_centroid": float(np.hypot(cx, cy)),
        "kernel_edge_mass": edge_mass,
        "ee_max_error": _encircled_energy_difference(matched, target),
        "l1_error": float(np.sum(np.abs(matched - target), dtype=np.float64)),
    }


def _passes_qc(metrics, config):
    return (
        metrics["kernel_min"] >= -1e-8
        and metrics["kernel_sum_error"] < config.kernel_sum_tolerance
        and metrics["kernel_centroid"] < config.kernel_centroid_tolerance
        and metrics["kernel_edge_mass"] < config.kernel_edge_tolerance
        and metrics["ee_max_error"] < config.encircled_energy_tolerance
        and metrics["l1_error"] < config.l1_tolerance
    )


class _KernelOperator(LinearOperator):
    def __init__(self, source, kernel_shape, regularization, constraint_weight=100.0):
        self.source = np.asarray(source, dtype=float)
        self.kernel_shape = tuple(kernel_shape)
        self.regularization = float(regularization)
        self.constraint_weight = float(constraint_weight)
        self.output_size = self.source.size
        self.kernel_size = int(np.prod(self.kernel_shape))
        y, x = _centered_coordinates(self.kernel_shape)
        scale = max(self.kernel_shape) // 2 or 1
        self.x = (x / scale).ravel()
        self.y = (y / scale).ravel()
        self.plan = _FFTConvolutionPlan(self.source, self.kernel_shape)
        rows = self.output_size + self.kernel_size + 3
        super().__init__(dtype=np.dtype(float), shape=(rows, self.kernel_size))

    def _matvec(self, vector):
        kernel = np.asarray(vector).reshape(self.kernel_shape)
        science = self.plan.convolve(kernel).ravel()
        smooth = np.sqrt(self.regularization) * laplace(kernel, mode="constant").ravel()
        constraints = self.constraint_weight * np.array([
            np.sum(vector),
            np.dot(self.x, vector),
            np.dot(self.y, vector),
        ])
        return np.concatenate((science, smooth, constraints))

    def _rmatvec(self, vector):
        vector = np.asarray(vector)
        science = vector[:self.output_size].reshape(self.source.shape)
        smooth = vector[self.output_size:self.output_size + self.kernel_size].reshape(self.kernel_shape)
        constraints = vector[-3:]
        result = self.plan.adjoint(science).ravel()
        result += np.sqrt(self.regularization) * laplace(smooth, mode="constant").ravel()
        result += self.constraint_weight * (
            constraints[0] + constraints[1] * self.x + constraints[2] * self.y
        )
        return result


class _FFTConvolutionPlan:
    """Cache the fixed source FFT used by iterative kernel optimization."""

    def __init__(self, source, kernel_shape):
        self.source = np.asarray(source, dtype=float)
        self.kernel_shape = tuple(kernel_shape)
        self.full_shape = tuple(
            self.source.shape[index] + self.kernel_shape[index] - 1
            for index in range(2)
        )
        self.fft_shape = tuple(next_fast_len(size) for size in self.full_shape)
        self.source_fft = rfft2(self.source, s=self.fft_shape)

        self.adjoint_full_shape = tuple(2 * size - 1 for size in self.source.shape)
        self.adjoint_fft_shape = tuple(next_fast_len(size) for size in self.adjoint_full_shape)
        self.reversed_source_fft = rfft2(
            self.source[::-1, ::-1],
            s=self.adjoint_fft_shape,
        )

    def convolve(self, kernel):
        kernel = np.asarray(kernel, dtype=float)
        full = irfft2(
            self.source_fft * rfft2(kernel, s=self.fft_shape),
            s=self.fft_shape,
        )[:self.full_shape[0], :self.full_shape[1]]
        y0 = (kernel.shape[0] - 1) // 2
        x0 = (kernel.shape[1] - 1) // 2
        return full[y0:y0 + self.source.shape[0], x0:x0 + self.source.shape[1]]

    def adjoint(self, residual):
        residual = np.asarray(residual, dtype=float)
        full = irfft2(
            rfft2(residual, s=self.adjoint_fft_shape) * self.reversed_source_fft,
            s=self.adjoint_fft_shape,
        )[:self.adjoint_full_shape[0], :self.adjoint_full_shape[1]]
        y0 = (self.source.shape[0] - 1) // 2
        x0 = (self.source.shape[1] - 1) // 2
        same = full[y0:y0 + self.source.shape[0], x0:x0 + self.source.shape[1]]
        ky0 = (same.shape[0] - self.kernel_shape[0]) // 2
        kx0 = (same.shape[1] - self.kernel_shape[1]) // 2
        return same[ky0:ky0 + self.kernel_shape[0], kx0:kx0 + self.kernel_shape[1]]


class _EnsembleKernelOperator(LinearOperator):
    def __init__(self, sources, weights, kernel_shape, regularization, constraint_weight=100.0):
        self.sources = [np.asarray(source, dtype=float) for source in sources]
        self.weights = np.sqrt(np.asarray(weights, dtype=float))
        self.kernel_shape = tuple(kernel_shape)
        self.regularization = float(regularization)
        self.constraint_weight = float(constraint_weight)
        self.output_size = self.sources[0].size
        self.kernel_size = int(np.prod(self.kernel_shape))
        self.plans = [_FFTConvolutionPlan(source, self.kernel_shape) for source in self.sources]
        y, x = _centered_coordinates(self.kernel_shape)
        scale = max(self.kernel_shape) // 2 or 1
        self.x = (x / scale).ravel()
        self.y = (y / scale).ravel()
        rows = len(self.sources) * self.output_size + self.kernel_size + 3
        super().__init__(dtype=np.dtype(float), shape=(rows, self.kernel_size))

    def _matvec(self, vector):
        kernel = np.asarray(vector).reshape(self.kernel_shape)
        science = [
            weight * plan.convolve(kernel).ravel()
            for weight, plan in zip(self.weights, self.plans)
        ]
        smooth = np.sqrt(self.regularization) * laplace(kernel, mode="constant").ravel()
        constraints = self.constraint_weight * np.array([
            np.sum(vector),
            np.dot(self.x, vector),
            np.dot(self.y, vector),
        ])
        return np.concatenate((*science, smooth, constraints))

    def _rmatvec(self, vector):
        vector = np.asarray(vector)
        result = np.zeros(self.kernel_shape, dtype=float)
        offset = 0
        for weight, plan in zip(self.weights, self.plans):
            residual = vector[offset:offset + self.output_size].reshape(self.sources[0].shape)
            result += weight * plan.adjoint(residual)
            offset += self.output_size
        smooth = vector[offset:offset + self.kernel_size].reshape(self.kernel_shape)
        constraints = vector[-3:]
        result = result.ravel()
        result += np.sqrt(self.regularization) * laplace(smooth, mode="constant").ravel()
        result += self.constraint_weight * (
            constraints[0] + constraints[1] * self.x + constraints[2] * self.y
        )
        return result


def _solve_kernel(source, target, size, regularization):
    shape = (size, size)
    operator = _KernelOperator(source, shape, regularization)
    rhs = np.concatenate((
        np.asarray(target, dtype=float).ravel(),
        np.zeros(size * size, dtype=float),
        np.array([operator.constraint_weight, 0.0, 0.0]),
    ))
    fit = lsq_linear(
        operator,
        rhs,
        bounds=(0.0, np.inf),
        method="trf",
        tol=1e-8,
        lsmr_tol="auto",
        max_iter=200,
        verbose=0,
    )
    if not fit.success and fit.optimality > 1e-5:
        raise RuntimeError(f"bounded kernel optimizer did not converge: {fit.message}")
    kernel = np.maximum(fit.x.reshape(shape), 0.0)
    total = float(np.sum(kernel, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("bounded kernel optimizer returned an empty kernel")
    kernel /= total
    matched = operator.plan.convolve(kernel)
    matched = np.maximum(matched, 0.0)
    matched /= np.sum(matched, dtype=np.float64)
    return kernel, matched


def _solve_kernel_ensemble(sources, targets, weights, size, regularization):
    shape = (size, size)
    operator = _EnsembleKernelOperator(sources, weights, shape, regularization)
    rhs_parts = [
        np.sqrt(weight) * np.asarray(target, dtype=float).ravel()
        for weight, target in zip(weights, targets)
    ]
    rhs = np.concatenate((
        *rhs_parts,
        np.zeros(size * size, dtype=float),
        np.array([operator.constraint_weight, 0.0, 0.0]),
    ))
    fit = lsq_linear(
        operator,
        rhs,
        bounds=(0.0, np.inf),
        method="trf",
        tol=1e-8,
        lsmr_tol="auto",
        max_iter=200,
        verbose=0,
    )
    if not fit.success and fit.optimality > 1e-5:
        raise RuntimeError(f"bounded ensemble kernel optimizer did not converge: {fit.message}")
    kernel = np.maximum(fit.x.reshape(shape), 0.0)
    total = float(np.sum(kernel, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("bounded ensemble kernel optimizer returned an empty kernel")
    kernel /= total
    matched = []
    for plan in operator.plans:
        item = np.maximum(plan.convolve(kernel), 0.0)
        item /= np.sum(item, dtype=np.float64)
        matched.append(item)
    return kernel, matched


def _mixture_parameters(parameters):
    sigmas = np.cumsum(np.exp(parameters[:3]))
    logits = parameters[3:6] - np.max(parameters[3:6])
    weights = np.exp(logits)
    weights /= np.sum(weights)
    q = 0.35 + 0.65 / (1.0 + np.exp(-parameters[6]))
    theta = parameters[7]
    return sigmas, weights, q, theta


def _integrated_gaussian_mixture(shape, parameters, oversampling):
    sigmas, weights, q, theta = _mixture_parameters(parameters)
    y, x = _centered_coordinates(shape)
    offsets = (np.arange(oversampling) + 0.5) / oversampling - 0.5
    model = np.zeros(shape, dtype=float)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for oy in offsets:
        for ox in offsets:
            xx = x + ox
            yy = y + oy
            xr = cos_t * xx + sin_t * yy
            yr = -sin_t * xx + cos_t * yy
            for weight, sigma in zip(weights, sigmas):
                model += weight * np.exp(-0.5 * ((xr / sigma) ** 2 + (yr / (q * sigma)) ** 2))
    model /= oversampling ** 2
    model /= np.sum(model, dtype=np.float64)
    return model


class SafePointSpreadFunction:
    """Pipeline entry points for safe, positive-kernel PSF matching."""

    @staticmethod
    def _model_tree(image_set):
        if not hasattr(image_set, "_safe_psf_models"):
            image_set._safe_psf_models = {}
        return image_set._safe_psf_models

    @staticmethod
    def _match_tree(image_set):
        if not hasattr(image_set, "_psf_match"):
            image_set._psf_match = {}
        return image_set._psf_match

    @staticmethod
    def _ensemble_tree(image_set):
        if not hasattr(image_set, "_psf_ensembles"):
            image_set._psf_ensembles = {}
        return image_set._psf_ensembles

    @staticmethod
    def _failure_tree(image_set):
        if not hasattr(image_set, "_safe_psf_failures"):
            image_set._safe_psf_failures = {}
        return image_set._safe_psf_failures

    @classmethod
    def _external_model(cls, observatory, band, config):
        config = SafePSFConfig.from_value(config)
        product = ImageProductMetadata(
            product_kind="exposure",
            facility=str(observatory),
            channel=str(band),
        )
        request = PSFRequest(
            product=product,
            galaxy="external",
            observatory=str(observatory),
            band=str(band),
            image_shape=(1, 1),
            header=fits.Header(),
        )
        provider = ExternalPSFProvider(validator=_validate_psf)
        if not provider.match(request, config):
            return None
        return provider.resolve(request, config).representative

    @classmethod
    def _packaged_model(cls, observatory, band, config):
        path = PointSpreadFunction.get_predefined_model(observatory, band)
        if path is None:
            return None
        with fits.open(path, memmap=False) as hdul:
            index = 1 if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None else 0
            data = np.asarray(hdul[index].data, dtype=float)
            header = hdul[index].header
        scale = header.get("PSCALE") or header.get("PIXSCALE") or useful_functions.get_pixel_scale(header)
        return _validate_psf(
            data,
            float(scale),
            f"packaged:{Path(path).name}",
            edge_tolerance=config.psf_edge_mass_tolerance,
            negative_peak_tolerance=config.negative_peak_tolerance,
            negative_mass_tolerance=config.negative_mass_tolerance,
        )

    @staticmethod
    def _header_fwhm(header):
        value = PointSpreadFunction.header_fwhm(header)
        return float(value) if PointSpreadFunction._valid_fwhm(value) else None

    @classmethod
    def _analytic_model(cls, fwhm_arcsec, pixel_scale, config, source="header-gaussian"):
        sigma_native = float(fwhm_arcsec) / 2.355 / float(pixel_scale)
        scale = float(pixel_scale) / config.fit_oversampling
        sigma = sigma_native * config.fit_oversampling
        radius = max(3, int(np.ceil(6.0 * sigma)))
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        data = np.exp(-0.5 * (x * x + y * y) / sigma ** 2)
        return _validate_psf(
            data,
            scale,
            source,
            edge_tolerance=config.psf_edge_mass_tolerance,
            negative_peak_tolerance=config.negative_peak_tolerance,
            negative_mass_tolerance=config.negative_mass_tolerance,
        )

    @classmethod
    def _empirical_model(cls, image, header, config):
        pixel_scale = float(useful_functions.get_pixel_scale(header))
        header_fwhm = cls._header_fwhm(header)
        if header_fwhm is not None and header_fwhm / pixel_scale < config.undersampled_fwhm_pixels:
            return None
        _, median, std = sigma_clipped_stats(image, sigma=5.0)
        if not np.isfinite(std) or std <= 0:
            return None
        sources = None
        for threshold in (50.0, 25.0, 15.0, 8.0):
            finder = DAOStarFinder(
                fwhm=max(2.0, (header_fwhm or 3 * pixel_scale) / pixel_scale),
                threshold=threshold * std,
            )
            sources = finder(image - median)
            if sources is not None and len(sources) >= config.min_stars:
                break
        if sources is None or len(sources) < config.min_stars:
            return None
        size = int(np.clip(np.ceil(10.0 / pixel_scale), 15, 41))
        size += 1 - size % 2
        half = size // 2
        saturation = header.get("SATURATE") or header.get("SATLEVEL")
        stamps = []
        peak_column = "flux" if "flux" in sources.colnames else "peak"
        order = np.argsort(np.asarray(sources[peak_column], dtype=float))[::-1]
        for row in sources[order]:
            x0, y0 = float(row["xcentroid"]), float(row["ycentroid"])
            ix, iy = int(round(x0)), int(round(y0))
            if iy - half < 0 or ix - half < 0 or iy + half >= image.shape[0] or ix + half >= image.shape[1]:
                continue
            stamp = np.asarray(image[iy - half:iy + half + 1, ix - half:ix + half + 1], dtype=float)
            if not np.all(np.isfinite(stamp)) or (saturation is not None and np.max(stamp) >= float(saturation)):
                continue
            border = np.concatenate((stamp[0], stamp[-1], stamp[1:-1, 0], stamp[1:-1, -1]))
            stamp = stamp - np.median(border)
            positive = np.maximum(stamp, 0.0)
            total = np.sum(positive)
            if total <= 0:
                continue
            yy, xx = _centered_coordinates(stamp.shape)
            cx = float(np.sum(xx * positive) / total)
            cy = float(np.sum(yy * positive) / total)
            aligned = shift(stamp, (-cy, -cx), order=3, mode="constant", cval=0.0)
            aligned = np.maximum(aligned, 0.0)
            aligned /= np.sum(aligned, dtype=np.float64)
            stamps.append(aligned)
            if len(stamps) >= 100:
                break
        if len(stamps) < config.min_stars:
            return None
        stack = np.nanmedian(np.stack(stamps), axis=0)
        stack = np.maximum(stack, 0.0)
        stack /= np.sum(stack, dtype=np.float64)
        y, x = _centered_coordinates(stack.shape)
        sigma_guess = np.sqrt(max(np.sum((x * x + y * y) * stack) / 2.0, 0.3))
        initial = np.array([
            np.log(max(0.6, sigma_guess * 0.5)),
            np.log(max(0.3, sigma_guess * 0.5)),
            np.log(max(0.3, sigma_guess)),
            1.5,
            0.0,
            -1.0,
            1.5,
            0.0,
        ])
        fit = least_squares(
            lambda p: (_integrated_gaussian_mixture(stack.shape, p, config.fit_oversampling) - stack).ravel(),
            initial,
            loss="soft_l1",
            f_scale=max(float(np.nanmedian(stack[stack > 0])), 1e-6),
            max_nfev=400,
        )
        if not fit.success:
            return None
        high_size = stack.shape[0] * config.fit_oversampling + 1
        y_h, x_h = _centered_coordinates((high_size, high_size))
        sigmas, weights, q, theta = _mixture_parameters(fit.x)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = (cos_t * x_h + sin_t * y_h) / config.fit_oversampling
        yr = (-sin_t * x_h + cos_t * y_h) / config.fit_oversampling
        model = np.zeros_like(xr)
        for weight, sigma in zip(weights, sigmas):
            model += weight * np.exp(-0.5 * ((xr / sigma) ** 2 + (yr / (q * sigma)) ** 2))
        return _validate_psf(
            model,
            pixel_scale / config.fit_oversampling,
            "empirical-positive-mixture",
            edge_tolerance=config.psf_edge_mass_tolerance,
            negative_peak_tolerance=config.negative_peak_tolerance,
            negative_mass_tolerance=config.negative_mass_tolerance,
        )

    @classmethod
    def _provider_registry(cls, request, config, measurement_image, measurement_header):
        registry = PSFProviderRegistry.with_registered(validator=_validate_psf)
        registry.register(ExternalPSFProvider(validator=_validate_psf))

        def packaged(_request, _config):
            model = cls._packaged_model(_request.observatory, _request.band, _config)
            if model is None:
                raise ProviderUnavailable("no packaged full PSF")
            return ResolvedPSF(model, singleton_ensemble(model), "packaged-full-psf")

        registry.register(FunctionalPSFProvider(
            "packaged-full-psf",
            100,
            lambda candidate, _: not candidate.product.requires_output_effective_psf,
            packaged,
        ))

        def empirical(_request, _config):
            if measurement_image is None:
                raise ProviderUnavailable("measurement region is unavailable")
            model = cls._empirical_model(measurement_image, measurement_header, _config)
            if model is None:
                raise ProviderUnavailable("insufficient well-sampled isolated stars")
            return ResolvedPSF(model, singleton_ensemble(model), "empirical-positive-mixture")

        registry.register(FunctionalPSFProvider(
            "empirical-positive-mixture",
            50,
            lambda candidate, _: not candidate.product.requires_output_effective_psf,
            empirical,
        ))

        def analytic(_request, _config):
            fwhm = cls._header_fwhm(measurement_header)
            scale = useful_functions.get_pixel_scale(measurement_header)
            if fwhm is None or not PointSpreadFunction._valid_fwhm(scale):
                raise ProviderUnavailable("header FWHM or angular sampling is unavailable")
            model = cls._analytic_model(fwhm, float(scale), _config)
            return ResolvedPSF(model, singleton_ensemble(model), "header-analytic-prf")

        registry.register(FunctionalPSFProvider(
            "header-analytic-prf",
            10,
            lambda candidate, _: not candidate.product.requires_output_effective_psf,
            analytic,
        ))
        return registry

    @classmethod
    def extract(
        cls,
        image_data,
        header,
        galaxy_name,
        observatory,
        band,
        image_set,
        safe_psf_config=None,
        metadata_resolver=None,
        **_,
    ):
        config = SafePSFConfig.from_value(safe_psf_config)
        target = f"{galaxy_name}/{observatory}/{band}"
        failures = []
        measurement_image = np.asarray(image_data)
        measurement_header = header
        if config.measurement_region == "pretrim" and config.pretrim_fov is not None:
            try:
                measurement_image, measurement_header, _ = PointSpreadFunction.measurement_data(
                    image_data,
                    header,
                    galaxy_name,
                    observatory,
                    band,
                    metadata_resolver=metadata_resolver,
                    psfconv_config=PSFConvolutionConfig(
                        pretrim_fov=config.pretrim_fov,
                        psf_measurement_region="pretrim",
                    ),
                )
            except Exception as exc:
                failures.append(f"measurement region: {exc}")
                measurement_image = None

        product = (
            getattr(image_set, "product_metadata", {})
            .get(galaxy_name, {})
            .get(observatory, {})
            .get(band)
        )
        if product is None:
            product = ImageProductMetadata(
                facility=str(observatory),
                channel=str(band),
                source_path=(
                    getattr(image_set, "files", {})
                    .get(galaxy_name, {})
                    .get(observatory, {})
                    .get(band)
                ),
            )
        else:
            product = ImageProductMetadata.from_value(product)
        shape = tuple(map(int, np.shape(image_data)))
        tile_bounds = (0.0, 0.0, float(shape[1] - 1), float(shape[0] - 1))
        synthesis = config.mosaic_synthesis
        configured_bounds = getattr(synthesis, "tile_bounds", None)
        if configured_bounds:
            group_id = getattr(product, "product_group_id", None)
            selected = (
                configured_bounds.get(str(group_id))
                if group_id is not None
                else None
            )
            if selected is None:
                selected = configured_bounds.get(str(product.product_id))
            if selected is not None:
                tile_bounds = tuple(map(float, selected))
        request = PSFRequest(
            product=product,
            galaxy=str(galaxy_name),
            observatory=str(observatory),
            band=str(band),
            image_shape=shape,
            header=header,
            image_data=np.asarray(image_data),
            tile_bounds=tile_bounds,
            phase_sampling=config.phase_sampling,
        )
        try:
            resolved = cls._provider_registry(
                request,
                config,
                measurement_image,
                measurement_header,
            ).resolve(request, config)
        except Exception as exc:
            failures.append(str(exc))
            resolved = None
        if resolved is None:
            reason = "; ".join(failures) or "no reliable PSF source"
            cls._failure_tree(image_set)[target] = reason
            image_set.psf = (galaxy_name, observatory, band, np.nan)
            return AnalysisProduct(
                kind="psf",
                data={"fwhm_arcsec": np.nan},
                metadata={"strategy": "safe", "source": "unavailable", "failure": reason},
            )

        model = resolved.representative
        models = cls._model_tree(image_set)
        models.setdefault(galaxy_name, {}).setdefault(observatory, {})[band] = model
        cls._ensemble_tree(image_set).setdefault(galaxy_name, {}).setdefault(observatory, {})[
            band
        ] = resolved.ensemble
        cls._failure_tree(image_set).pop(target, None)
        image_set.psf = (galaxy_name, observatory, band, model.fwhm_arcsec)
        emit_detail(
            f"Safe PSF model {target}: source={model.source}, scale={model.pixel_scale_arcsec:.6g}, "
            f"fwhm={model.fwhm_arcsec:.6g}, r80={model.r80_arcsec:.6g} arcsec"
        )
        return AnalysisProduct(
            kind="psf",
            data={"fwhm_arcsec": model.fwhm_arcsec},
            metadata={
                "strategy": "safe",
                "source": model.source,
                "provider": resolved.provider,
                "diagnostics": resolved.diagnostics,
                **model.quality,
            },
        )

    @classmethod
    def _match_one(cls, model, target_model, native_sampling, config):
        native_sampling = PixelSampling.from_value(native_sampling)
        source = _resample_model(model, native_sampling)
        target = _resample_model(target_model, native_sampling)
        source, target = _common_shape(source, target)
        source /= np.sum(source, dtype=np.float64)
        target /= np.sum(target, dtype=np.float64)
        if np.sum(np.abs(source - target)) < min(config.l1_tolerance, 1e-8):
            kernel = np.ones((1, 1), dtype=np.float32)
            matched = source.copy()
            metrics = _kernel_metrics(kernel, matched, target)
            return kernel, matched, target, metrics

        variance_source = np.sum(np.square(_centered_coordinates(source.shape)[0]) * source)
        variance_target = np.sum(np.square(_centered_coordinates(target.shape)[0]) * target)
        sigma_guess = np.sqrt(max(variance_target - variance_source, 0.25))
        initial_size = max(3, int(2 * np.ceil(4.0 * sigma_guess) + 1))
        max_size = min(config.max_kernel_size, min(source.shape))
        if max_size % 2 == 0:
            max_size -= 1
        initial_size = min(initial_size, max_size)
        if initial_size % 2 == 0:
            initial_size += 1
        best = None
        for size in range(initial_size, max_size + 1, 4):
            if size % 2 == 0:
                size += 1
            for regularization in config.kernel_regularization:
                kernel, matched = _solve_kernel(source, target, size, regularization)
                metrics = _kernel_metrics(kernel, matched, target)
                metrics["regularization"] = float(regularization)
                metrics["kernel_size"] = float(size)
                if best is None or metrics["l1_error"] < best[3]["l1_error"]:
                    best = (kernel, matched, target, metrics)
                if _passes_qc(metrics, config):
                    return kernel, matched, target, metrics
        assert best is not None
        raise RuntimeError(
            "positive kernel did not pass QC: "
            + ", ".join(f"{key}={value:.4g}" for key, value in best[3].items())
        )

    @classmethod
    def _match_ensemble(
        cls,
        ensemble,
        target_model,
        native_sampling,
        config,
        target_samples=None,
    ):
        if len(ensemble.samples) == 1:
            return cls._match_one(
                ensemble.samples[0].model,
                target_model if target_samples is None else target_samples[0],
                native_sampling,
                config,
            )

        source_arrays = [
            _resample_model(sample.model, native_sampling)
            for sample in ensemble.samples
        ]
        if target_samples is None:
            target_arrays = [
                _resample_model(target_model, native_sampling)
                for _ in ensemble.samples
            ]
        else:
            if len(target_samples) != len(ensemble.samples):
                raise ValueError("Target PSF ensemble does not match the source ensemble")
            target_arrays = [
                _resample_model(model, native_sampling)
                for model in target_samples
            ]
        original_ensemble_size = len(source_arrays)
        original_weights = np.asarray([sample.weight for sample in ensemble.samples], dtype=float)
        grouped = {}
        for array, target_array, weight in zip(source_arrays, target_arrays, original_weights):
            digest = sha256()
            digest.update(np.ascontiguousarray(array).view(np.uint8))
            digest.update(np.ascontiguousarray(target_array).view(np.uint8))
            key = digest.hexdigest()
            if key in grouped:
                grouped[key][2] += float(weight)
            else:
                grouped[key] = [array, target_array, float(weight)]
        source_arrays = [item[0] for item in grouped.values()]
        target_arrays = [item[1] for item in grouped.values()]
        grouped_weights = np.asarray([item[2] for item in grouped.values()], dtype=float)
        ny = max([*(item.shape[0] for item in target_arrays), *(item.shape[0] for item in source_arrays)])
        nx = max([*(item.shape[1] for item in target_arrays), *(item.shape[1] for item in source_arrays)])
        ny += 1 - ny % 2
        nx += 1 - nx % 2
        shape = (ny, nx)
        sources = [_pad_center(item, shape) for item in source_arrays]
        targets = [_pad_center(item, shape) for item in target_arrays]
        sources = [item / np.sum(item, dtype=np.float64) for item in sources]
        targets = [item / np.sum(item, dtype=np.float64) for item in targets]
        weights = grouped_weights
        weights /= np.sum(weights)

        initial_l1 = [float(np.sum(np.abs(source - target))) for source, target in zip(sources, targets)]
        if max(initial_l1) < min(config.l1_tolerance, 1e-8):
            kernel = np.ones((1, 1), dtype=np.float32)
            metrics = _kernel_metrics(kernel, sources[0], targets[0])
            metrics.update({
                "l1_error": max(initial_l1),
                "ee_max_error": max(
                    _encircled_energy_difference(source, target)
                    for source, target in zip(sources, targets)
                ),
                "ensemble_size": float(len(sources)),
                "ensemble_samples": float(original_ensemble_size),
            })
            matched_mean = sum(weight * item for weight, item in zip(weights, sources))
            target_mean = sum(weight * item for weight, item in zip(weights, targets))
            return kernel, matched_mean, target_mean, metrics

        source_mean = sum(weight * item for weight, item in zip(weights, sources))
        variance_source = np.sum(np.square(_centered_coordinates(shape)[0]) * source_mean)
        variance_target = np.sum(np.square(_centered_coordinates(shape)[0]) * targets[0])
        sigma_guess = np.sqrt(max(variance_target - variance_source, 0.25))
        initial_size = max(3, int(2 * np.ceil(4.0 * sigma_guess) + 1))
        max_size = min(config.max_kernel_size, min(shape))
        if max_size % 2 == 0:
            max_size -= 1
        initial_size = min(initial_size, max_size)
        if initial_size % 2 == 0:
            initial_size += 1
        best = None
        for size in range(initial_size, max_size + 1, 4):
            if size % 2 == 0:
                size += 1
            for regularization in config.kernel_regularization:
                kernel, matched_items = _solve_kernel_ensemble(
                    sources,
                    targets,
                    weights,
                    size,
                    regularization,
                )
                per_sample = [
                    _kernel_metrics(kernel, matched, target)
                    for matched, target in zip(matched_items, targets)
                ]
                worst_l1_index = int(np.argmax([item["l1_error"] for item in per_sample]))
                metrics = dict(per_sample[worst_l1_index])
                metrics["l1_error"] = max(item["l1_error"] for item in per_sample)
                metrics["ee_max_error"] = max(item["ee_max_error"] for item in per_sample)
                metrics["kernel_min"] = min(item["kernel_min"] for item in per_sample)
                metrics["kernel_sum_error"] = max(item["kernel_sum_error"] for item in per_sample)
                metrics["kernel_centroid"] = max(item["kernel_centroid"] for item in per_sample)
                metrics["kernel_edge_mass"] = max(item["kernel_edge_mass"] for item in per_sample)
                metrics["regularization"] = float(regularization)
                metrics["kernel_size"] = float(size)
                metrics["ensemble_size"] = float(len(sources))
                metrics["ensemble_samples"] = float(original_ensemble_size)
                metrics["worst_sample"] = float(worst_l1_index)
                matched_mean = sum(
                    weight * item for weight, item in zip(weights, matched_items)
                )
                target_mean = sum(weight * item for weight, item in zip(weights, targets))
                candidate = (kernel, matched_mean, target_mean, metrics)
                if best is None or metrics["l1_error"] < best[3]["l1_error"]:
                    best = candidate
                if _passes_qc(metrics, config):
                    return candidate
        assert best is not None
        raise RuntimeError(
            "positive ensemble kernel did not pass worst-case QC: "
            + ", ".join(
                f"{key}={value:.4g}"
                for key, value in best[3].items()
                if isinstance(value, (int, float, np.number))
            )
        )

    @staticmethod
    def _phase_key(phase):
        if phase is None:
            return None
        return tuple(round(float(value), 8) for value in phase)

    @classmethod
    def _target_samples(cls, source_ensemble, base_ensemble, sigma_arcsec, config):
        groups = {}
        for sample in base_ensemble.samples:
            groups.setdefault(cls._phase_key(sample.phase), []).append(sample)
        rendered = {}
        for phase, samples in groups.items():
            ny = max(sample.model.data.shape[0] for sample in samples)
            nx = max(sample.model.data.shape[1] for sample in samples)
            ny += 1 - ny % 2
            nx += 1 - nx % 2
            weights = np.asarray([sample.weight for sample in samples], dtype=float)
            weights /= np.sum(weights)
            data = sum(
                weight * _pad_center(sample.model.data, (ny, nx))
                for weight, sample in zip(weights, samples)
            )
            scale = samples[0].model.pixel_scale_arcsec
            if sigma_arcsec > 0:
                padding = int(np.ceil(6.0 * sigma_arcsec / scale))
                data = gaussian_filter(
                    np.pad(data, padding),
                    sigma_arcsec / scale,
                    mode="constant",
                    cval=0.0,
                    truncate=6.0,
                )
            data = np.maximum(data, 0.0)
            data /= np.sum(data, dtype=np.float64)
            rendered[phase] = _validate_psf(
                data,
                samples[0].model.sampling,
                f"target-phase:{phase}+gaussian({sigma_arcsec:.6g}arcsec)",
                edge_tolerance=max(config.psf_edge_mass_tolerance, 1e-3),
                negative_peak_tolerance=config.negative_peak_tolerance,
                negative_mass_tolerance=config.negative_mass_tolerance,
                frame=samples[0].model.frame,
            )
        fallback = rendered.get(None)
        if fallback is None:
            fallback = next(iter(rendered.values()))
        return [rendered.get(cls._phase_key(sample.phase), fallback) for sample in source_ensemble.samples]

    @classmethod
    def _attempt_target(
        cls,
        image_set,
        galaxy,
        models,
        base,
        base_ensemble,
        sigma_arcsec,
        config,
    ):
        if sigma_arcsec > 0:
            padding = int(np.ceil(6.0 * sigma_arcsec / base.pixel_scale_arcsec))
            base_data = np.pad(np.asarray(base.data, dtype=float), padding)
            target_data = gaussian_filter(
                base_data,
                sigma_arcsec / base.pixel_scale_arcsec,
                mode="constant",
                cval=0.0,
                truncate=6.0,
            )
        else:
            target_data = np.asarray(base.data, dtype=float).copy()
        target_data = np.maximum(target_data, 0.0)
        target_data /= np.sum(target_data, dtype=np.float64)
        target_model = _validate_psf(
            target_data,
            base.pixel_scale_arcsec,
            f"target:{base.source}+gaussian({sigma_arcsec:.6g}arcsec)",
            edge_tolerance=max(config.kernel_edge_tolerance, 1e-3),
        )
        results = {}
        failures = {}
        ensemble_tree = cls._ensemble_tree(image_set).get(galaxy, {})
        for observatory, bands in models.items():
            for band, model in bands.items():
                key = f"{galaxy}/{observatory}/{band}"
                try:
                    header = image_set.header[galaxy][observatory][band]
                    native_scale = float(useful_functions.get_pixel_scale(header))
                    product = (
                        getattr(image_set, "product_metadata", {})
                        .get(galaxy, {})
                        .get(observatory, {})
                        .get(band)
                    )
                    native_sampling = (
                        ImageProductMetadata.from_value(product).sampling
                        if product is not None
                        else None
                    )
                    if native_sampling is None:
                        native_sampling = PixelSampling.isotropic(native_scale)
                    ensemble = ensemble_tree.get(observatory, {}).get(band)
                    if ensemble is None:
                        ensemble = singleton_ensemble(model)
                    target_samples = cls._target_samples(
                        ensemble,
                        base_ensemble,
                        sigma_arcsec,
                        config,
                    )
                    kernel, matched, target_native, metrics = cls._match_ensemble(
                        ensemble,
                        target_model,
                        native_sampling,
                        config,
                        target_samples=target_samples,
                    )
                    target_native_model = EffectivePSF(
                        np.asarray(target_native, dtype=np.float32),
                        native_scale,
                        target_model.fwhm_arcsec,
                        target_model.r50_arcsec,
                        target_model.r80_arcsec,
                        target_model.source,
                        frame="image",
                        sampling=native_sampling,
                        provenance={"ensemble_size": len(ensemble.samples)},
                    )
                    results.setdefault(observatory, {})[band] = PSFMatchResult(
                        input_psf=model,
                        target_psf=target_native_model,
                        kernel=np.asarray(kernel, dtype=np.float32),
                        matched_psf=np.asarray(matched, dtype=np.float32),
                        metrics=metrics,
                    )
                except Exception as exc:
                    failures[key] = str(exc)
        return results, failures

    @classmethod
    def prepare(cls, image_set, safe_psf_config=None, **_):
        config = SafePSFConfig.from_value(safe_psf_config)
        all_models = cls._model_tree(image_set)
        missing = dict(cls._failure_tree(image_set))
        for galaxy, observatories in image_set.data.items():
            for observatory, bands in observatories.items():
                for band in bands:
                    if band not in all_models.get(galaxy, {}).get(observatory, {}):
                        missing.setdefault(
                            f"{galaxy}/{observatory}/{band}", "PSF model was not extracted"
                        )
        if missing:
            raise PSFMatchError(missing)

        legacy_groups = {}
        for galaxy, observatories in all_models.items():
            for observatory, bands in observatories.items():
                for band, model in bands.items():
                    if not model.provenance.get("legacy_selector"):
                        continue
                    legacy_groups.setdefault(
                        (str(observatory).lower(), str(band).lower()), []
                    ).append((galaxy, observatory, band, model.provenance.get("resolved_product_id")))
        ambiguous_legacy = {}
        for entries in legacy_groups.values():
            if len({entry[3] for entry in entries}) <= 1:
                continue
            for galaxy, observatory, band, _ in entries:
                ambiguous_legacy[f"{galaxy}/{observatory}/{band}"] = (
                    "legacy PSF manifest selector is ambiguous across multiple image products; "
                    "use a schema-v2 product_id selector"
                )
        if ambiguous_legacy:
            raise PSFMatchError(ambiguous_legacy)

        prepared = {}
        summaries = {}
        failures = {}
        for galaxy, models in all_models.items():
            flattened = [
                (observatory, band, model)
                for observatory, bands in models.items()
                for band, model in bands.items()
            ]
            base_observatory, base_band, base = max(
                flattened,
                key=lambda item: item[2].r80_arcsec,
            )
            base_ensemble = (
                cls._ensemble_tree(image_set)
                .get(galaxy, {})
                .get(base_observatory, {})
                .get(base_band, singleton_ensemble(base))
            )
            last_failure_sigma = 0.0
            success_sigma = None
            success_results = None
            trial_failures = None
            for trial in range(config.max_target_trials):
                sigma = 0.0 if trial == 0 else base.fwhm_arcsec * 0.05 * 1.5 ** (trial - 1)
                results, current_failures = cls._attempt_target(
                    image_set, galaxy, models, base, base_ensemble, sigma, config
                )
                if not current_failures:
                    success_sigma = sigma
                    success_results = results
                    break
                last_failure_sigma = sigma
                trial_failures = current_failures
            if success_results is None:
                failures.update(trial_failures or {galaxy: "target search failed"})
                continue
            if success_sigma and config.target_bisection_steps:
                low, high = last_failure_sigma, success_sigma
                for _ in range(config.target_bisection_steps):
                    middle = 0.5 * (low + high)
                    results, current_failures = cls._attempt_target(
                        image_set, galaxy, models, base, base_ensemble, middle, config
                    )
                    if current_failures:
                        low = middle
                    else:
                        high = middle
                        success_sigma = middle
                        success_results = results
            prepared[galaxy] = success_results
            summaries[galaxy] = {
                "base_source": base.source,
                "safety_sigma_arcsec": float(success_sigma or 0.0),
                "bands": sum(len(bands) for bands in success_results.values()),
            }
        if failures:
            raise PSFMatchError(failures)
        image_set._psf_match = prepared
        for galaxy, observatories in prepared.items():
            for observatory, bands in observatories.items():
                for band, result in bands.items():
                    emit_detail(
                        f"Safe PSF kernel {galaxy}/{observatory}/{band}: shape={result.kernel.shape}, "
                        f"target={result.target_psf.fwhm_arcsec:.6g} arcsec, "
                        f"L1={result.metrics['l1_error']:.4g}, EE={result.metrics['ee_max_error']:.4g}"
                    )
        return AnalysisProduct(kind="psf_match", data=summaries, metadata={"strategy": "safe"})

    @classmethod
    def pretrim(
        cls,
        image_data,
        header,
        error_data,
        galaxy_name,
        observatory,
        band,
        image_set,
        safe_psf_config=None,
        metadata_resolver=None,
    ):
        config = SafePSFConfig.from_value(safe_psf_config)
        if config.pretrim_fov is None:
            return
        result = cls._match_tree(image_set)[galaxy_name][observatory][band]
        pixel_scale = float(useful_functions.get_pixel_scale(header))
        halo = max(result.kernel.shape[0] // 2, result.kernel.shape[1] // 2) * pixel_scale * u.arcsec
        requested = tuple(value + 2 * halo for value in config.pretrim_fov)
        center = useful_functions.get_sky_loc(
            galaxy_name,
            header=header,
            metadata_resolver=metadata_resolver,
            required=True,
            image_shape=np.shape(image_data),
        )
        from .registration import Register
        try:
            trim_image, trim_header, trim_error, _ = Register._trim_sky_with_info(
                image=image_data,
                header=header,
                error=error_data,
                skycoord=center,
                size=requested,
                mode="partial",
                fill_value=np.nan,
                context=f"safe PSF pretrim for {galaxy_name}/{observatory}/{band}",
            )
        except NoOverlapError as exc:
            raise ValueError(f"Safe PSF trim does not overlap {galaxy_name}/{observatory}/{band}") from exc
        image_set.update_data(trim_image, galaxy_name, observatory, band)
        image_set.update_header(trim_header, galaxy_name, observatory, band)
        if error_data is not None:
            image_set.update_error(trim_error, galaxy_name, observatory, band)

    @staticmethod
    def _strict_convolve(engine, array, kernel, tolerance):
        array = np.asarray(array)
        finite = np.isfinite(array)
        clean = np.where(finite, array, 0.0).astype(np.float32, copy=False)
        result = engine.convolve(clean, kernel)
        support = engine.convolve(finite.astype(np.float32), kernel)
        result[support < 1.0 - tolerance] = np.nan
        return result

    @classmethod
    def convolution(
        cls,
        image_data,
        header,
        error_data,
        galaxy_name,
        observatory,
        band,
        image_set,
        convolution_engine=None,
        safe_psf_config=None,
        metadata_resolver=None,
        **_,
    ):
        config = SafePSFConfig.from_value(safe_psf_config)
        result = cls._match_tree(image_set)[galaxy_name][observatory][band]
        engine = convolution_engine or PSFConvolutionEngine(config.convolution_config())
        kernel = np.asarray(result.kernel, dtype=np.float32)
        science = cls._strict_convolve(engine, image_data, kernel, config.support_tolerance)
        uncertainty = None
        if error_data is not None:
            error = np.asarray(error_data)
            if error.shape != np.shape(image_data):
                raise ValueError("Science and STD error map shapes do not match")
            if np.any(np.isfinite(error) & (error < 0)):
                raise ValueError("STD error map contains finite negative values")
            finite = np.isfinite(error) & np.isfinite(image_data)
            variance = np.where(finite, error.astype(float) ** 2, 0.0).astype(np.float32)
            convolved_variance = engine.convolve(variance, kernel ** 2)
            support = engine.convolve(finite.astype(np.float32), kernel)
            uncertainty = np.sqrt(np.maximum(convolved_variance, 0.0)).astype(np.float32)
            uncertainty[support < 1.0 - config.support_tolerance] = np.nan
        updated_header = header.copy()
        updated_header["PSFMODE"] = ("SAFE", "Spec7DT PSF matching strategy")
        updated_header["PSFIN"] = (result.input_psf.fwhm_arcsec, "Input PSF FWHM [arcsec]")
        updated_header["PSFOUT"] = (result.target_psf.fwhm_arcsec, "Matched PSF FWHM [arcsec]")
        updated_header["PSFSRC"] = (result.input_psf.source[:68], "Input PSF model source")
        updated_header["PSFEE"] = (result.metrics["ee_max_error"], "PSF encircled-energy error")
        updated_header["PSFL1"] = (result.metrics["l1_error"], "Normalized PSF L1 error")
        updated_header["PSFMATCH"] = (True, "Image was PSF matched")
        if config.pretrim_fov is not None:
            center = useful_functions.get_sky_loc(
                galaxy_name,
                header=updated_header,
                metadata_resolver=metadata_resolver,
                required=True,
                image_shape=science.shape,
            )
            from .registration import Register

            science, updated_header, uncertainty, _ = Register._trim_sky_with_info(
                image=science,
                header=updated_header,
                error=uncertainty,
                skycoord=center,
                size=config.pretrim_fov,
                mode="partial",
                fill_value=np.nan,
                context=f"safe PSF valid crop for {galaxy_name}/{observatory}/{band}",
            )
        image_set.update_data(science, galaxy_name, observatory, band)
        image_set.update_header(updated_header, galaxy_name, observatory, band)
        if uncertainty is not None:
            image_set.update_error(uncertainty, galaxy_name, observatory, band)


def _write_sampling_header(header, model):
    matrix = np.asarray(model.sampling.matrix_arcsec, dtype=float)
    header["PSFM11"] = float(matrix[0, 0])
    header["PSFM12"] = float(matrix[0, 1])
    header["PSFM21"] = float(matrix[1, 0])
    header["PSFM22"] = float(matrix[1, 1])
    header["PSFFRAME"] = str(model.sampling.frame)


def _sampling_from_header(header):
    matrix_keys = ("PSFM11", "PSFM12", "PSFM21", "PSFM22")
    if all(key in header for key in matrix_keys):
        from ..psf.models import PixelSampling

        return PixelSampling(
            (
                (float(header["PSFM11"]), float(header["PSFM12"])),
                (float(header["PSFM21"]), float(header["PSFM22"])),
            ),
            frame=str(header.get("PSFFRAME", "image")),
        )
    return None


def write_psf_artifact(path, model, result=None, ensemble=None):
    """Persist one safe PSF model/match without placing arrays in JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["PIXSCALE"] = model.pixel_scale_arcsec
    header["FWHM"] = model.fwhm_arcsec
    header["R50"] = model.r50_arcsec
    header["R80"] = model.r80_arcsec
    header["PSFSRC"] = model.source[:68]
    _write_sampling_header(header, model)
    hdus = [fits.PrimaryHDU(np.asarray(model.data, dtype=np.float32), header=header)]
    if ensemble is not None:
        for index, sample in enumerate(ensemble.samples, start=1):
            sample_header = fits.Header()
            sample_header["EXTVER"] = int(index)
            sample_header["WEIGHT"] = float(sample.weight)
            if sample.position is not None:
                sample_header["POSX"] = float(sample.position[0])
                sample_header["POSY"] = float(sample.position[1])
            if sample.phase is not None:
                sample_header["PHASEX"] = float(sample.phase[1])
                sample_header["PHASEY"] = float(sample.phase[0])
            sample_header["PIXSCALE"] = sample.model.pixel_scale_arcsec
            sample_header["FWHM"] = sample.model.fwhm_arcsec
            sample_header["R50"] = sample.model.r50_arcsec
            sample_header["R80"] = sample.model.r80_arcsec
            sample_header["PSFSRC"] = sample.model.source[:68]
            _write_sampling_header(sample_header, sample.model)
            hdus.append(fits.ImageHDU(
                np.asarray(sample.model.data, dtype=np.float32),
                header=sample_header,
                name="SAMPLE",
            ))
    if result is not None:
        target_header = fits.Header()
        target_header["PIXSCALE"] = result.target_psf.pixel_scale_arcsec
        target_header["FWHM"] = result.target_psf.fwhm_arcsec
        target_header["R50"] = result.target_psf.r50_arcsec
        target_header["R80"] = result.target_psf.r80_arcsec
        target_header["PSFSRC"] = result.target_psf.source[:68]
        _write_sampling_header(target_header, result.target_psf)
        metric_header = fits.Header()
        for key, name in (
            ("l1_error", "QCL1"),
            ("ee_max_error", "QCEE"),
            ("kernel_edge_mass", "QCEDGE"),
            ("kernel_centroid", "QCCENT"),
            ("kernel_min", "QCMIN"),
            ("kernel_sum_error", "QCSUM"),
            ("regularization", "REGUL"),
            ("kernel_size", "KERNSIZE"),
        ):
            if key in result.metrics:
                metric_header[name] = float(result.metrics[key])
        hdus.extend([
            fits.ImageHDU(np.asarray(result.target_psf.data, dtype=np.float32), header=target_header, name="TARGET"),
            fits.ImageHDU(np.asarray(result.kernel, dtype=np.float32), header=metric_header, name="KERNEL"),
            fits.ImageHDU(np.asarray(result.matched_psf, dtype=np.float32), name="MATCHED"),
        ])
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=True)


def read_psf_artifact(path, *, include_ensemble=False):
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        model = EffectivePSF(
            np.asarray(hdul[0].data, dtype=np.float32),
            float(header["PIXSCALE"]),
            float(header["FWHM"]),
            float(header["R50"]),
            float(header["R80"]),
            str(header.get("PSFSRC", "checkpoint")),
            sampling=_sampling_from_header(header),
        )
        samples = []
        if include_ensemble:
            for hdu in hdul:
                if str(hdu.header.get("EXTNAME", "")).upper() != "SAMPLE":
                    continue
                sh = hdu.header
                sample_model = EffectivePSF(
                    np.asarray(hdu.data, dtype=np.float32),
                    float(sh["PIXSCALE"]),
                    float(sh["FWHM"]),
                    float(sh["R50"]),
                    float(sh["R80"]),
                    str(sh.get("PSFSRC", "checkpoint-sample")),
                    sampling=_sampling_from_header(sh),
                )
                position = (
                    (float(sh["POSX"]), float(sh["POSY"]))
                    if "POSX" in sh and "POSY" in sh else None
                )
                phase = (
                    (float(sh["PHASEY"]), float(sh["PHASEX"]))
                    if "PHASEX" in sh and "PHASEY" in sh else None
                )
                samples.append(PSFSample(
                    sample_model,
                    position=position,
                    phase=phase,
                    weight=float(sh.get("WEIGHT", 1.0)),
                ))
        ensemble = PSFEnsemble(samples) if samples else singleton_ensemble(model)
        if "TARGET" not in hdul or "KERNEL" not in hdul or "MATCHED" not in hdul:
            return (model, None, ensemble) if include_ensemble else (model, None)
        target_header = hdul["TARGET"].header
        target = EffectivePSF(
            np.asarray(hdul["TARGET"].data, dtype=np.float32),
            float(target_header["PIXSCALE"]),
            float(target_header["FWHM"]),
            float(target_header["R50"]),
            float(target_header["R80"]),
            str(target_header.get("PSFSRC", "checkpoint-target")),
            sampling=_sampling_from_header(target_header),
        )
        kh = hdul["KERNEL"].header
        metric_names = {
            "QCL1": "l1_error",
            "QCEE": "ee_max_error",
            "QCEDGE": "kernel_edge_mass",
            "QCCENT": "kernel_centroid",
            "QCMIN": "kernel_min",
            "QCSUM": "kernel_sum_error",
            "REGUL": "regularization",
            "KERNSIZE": "kernel_size",
        }
        metrics = {name: float(kh[key]) for key, name in metric_names.items() if key in kh}
        result = PSFMatchResult(
            model,
            target,
            np.asarray(hdul["KERNEL"].data, dtype=np.float32),
            np.asarray(hdul["MATCHED"].data, dtype=np.float32),
            metrics,
        )
    return (model, result, ensemble) if include_ensemble else (model, result)


__all__ = [
    "EffectivePSF",
    "PSFMatchError",
    "PSFMatchResult",
    "SafePSFConfig",
    "SafePointSpreadFunction",
    "read_psf_artifact",
    "write_psf_artifact",
]
