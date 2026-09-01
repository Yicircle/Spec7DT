from dataclasses import dataclass
from importlib import resources
from time import perf_counter
from typing import Literal
import warnings

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.nddata import NDData
from astropy.nddata.utils import NoOverlapError
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyWarning
from scipy.signal import oaconvolve

from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars, EPSFBuilder, EPSFFitter, fit_2dgaussian

warnings.simplefilter('ignore', category=AstropyWarning)


from ..utils.utility import useful_functions
from ..utils.reporting import emit_alert, emit_detail
from ..handlers.filter_handler import Filters
from ..utils.file_handler import Parsers


def _normalize_angular_fov(value):
    if value is None:
        return None

    if isinstance(value, u.Quantity) and value.isscalar:
        values = (value, value)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise TypeError(
                "pretrim_fov must be an angular Quantity or a (ny, nx) pair of "
                "angular Quantities"
            ) from exc
        if len(values) != 2:
            raise ValueError("pretrim_fov must contain exactly two values in (ny, nx) order")

    normalized = []
    for axis, item in zip(("ny", "nx"), values):
        if not isinstance(item, u.Quantity) or not item.unit.is_equivalent(u.arcsec):
            raise TypeError(f"pretrim_fov {axis} must be an angular Quantity")
        item = item.to(u.arcsec)
        if not item.isscalar or not np.isfinite(item.value) or item.value <= 0:
            raise ValueError(f"pretrim_fov {axis} must be a finite positive angular size")
        normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class PSFConvolutionConfig:
    """Configuration for native-grid PSF matching and its optional pre-trim."""

    pretrim_fov: u.Quantity | tuple[u.Quantity, u.Quantity] | None = None
    psf_measurement_region: Literal["pretrim", "full"] = "pretrim"
    use_gpu: bool = False
    gpu_fallback: Literal["cpu", "error"] = "cpu"
    kernel_truncate: float = 4.0
    min_epsf_stars: int = 20

    def __post_init__(self):
        if not isinstance(self.use_gpu, (bool, np.bool_)):
            raise TypeError("use_gpu must be a boolean")
        if self.psf_measurement_region not in {"pretrim", "full"}:
            raise ValueError("psf_measurement_region must be 'pretrim' or 'full'")
        if self.gpu_fallback not in {"cpu", "error"}:
            raise ValueError("gpu_fallback must be 'cpu' or 'error'")
        if not np.isfinite(self.kernel_truncate) or self.kernel_truncate <= 0:
            raise ValueError("kernel_truncate must be finite and positive")
        if (
            not isinstance(self.min_epsf_stars, (int, np.integer))
            or isinstance(self.min_epsf_stars, (bool, np.bool_))
            or self.min_epsf_stars < 1
        ):
            raise ValueError("min_epsf_stars must be a positive integer")
        object.__setattr__(self, "pretrim_fov", _normalize_angular_fov(self.pretrim_fov))
        object.__setattr__(self, "use_gpu", bool(self.use_gpu))
        object.__setattr__(self, "kernel_truncate", float(self.kernel_truncate))
        object.__setattr__(self, "min_epsf_stars", int(self.min_epsf_stars))

    @classmethod
    def from_value(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("psfconv_config must be PSFConvolutionConfig, dict, or None")


class PSFConvolutionEngine:
    """Stable CPU/GPU overlap-add backend selected once per pipeline run."""

    def __init__(self, config=None):
        self.config = PSFConvolutionConfig.from_value(config)
        self.backend = "cpu"
        self._cupy = None
        self._gpu_oaconvolve = None
        self._gpu_kernel_cache = {}
        self._gpu_warning_emitted = False
        if self.config.use_gpu:
            self._initialize_gpu()

    def _initialize_gpu(self):
        try:
            import cupy as cp
            from cupyx.scipy.signal import oaconvolve as gpu_oaconvolve

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA-capable device is available")
            cp.cuda.runtime.memGetInfo()
        except Exception as exc:
            self._handle_gpu_failure(exc, during_runtime=False)
            return

        self._cupy = cp
        self._gpu_oaconvolve = gpu_oaconvolve
        self.backend = "gpu"

    def _handle_gpu_failure(self, exc, *, during_runtime):
        context = "during convolution" if during_runtime else "during initialization"
        message = f"GPU PSF convolution failed {context}: {exc}"
        if self.config.gpu_fallback == "error":
            raise RuntimeError(message) from exc
        if not self._gpu_warning_emitted:
            warnings.warn(f"{message}; using CPU for this and all remaining images.", RuntimeWarning)
            self._gpu_warning_emitted = True
        if self._cupy is not None:
            try:
                self._cupy.get_default_memory_pool().free_all_blocks()
                self._cupy.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
        self.backend = "cpu"
        self._gpu_kernel_cache.clear()
        self._cupy = None
        self._gpu_oaconvolve = None

    @staticmethod
    def _kernel_cache_key(kernel):
        contiguous = np.ascontiguousarray(kernel, dtype=np.float32)
        return contiguous.shape, contiguous.tobytes()

    def convolve(self, array, kernel):
        array = np.ascontiguousarray(array, dtype=np.float32)
        kernel = np.ascontiguousarray(kernel, dtype=np.float32)
        if self.backend == "gpu":
            try:
                cp = self._cupy
                key = self._kernel_cache_key(kernel)
                kernel_gpu = self._gpu_kernel_cache.get(key)
                if kernel_gpu is None:
                    kernel_gpu = cp.asarray(kernel)
                    self._gpu_kernel_cache[key] = kernel_gpu
                result = self._gpu_oaconvolve(cp.asarray(array), kernel_gpu, mode="same")
                return cp.asnumpy(result).astype(np.float32, copy=False)
            except Exception as exc:
                self._handle_gpu_failure(exc, during_runtime=True)

        return np.asarray(oaconvolve(array, kernel, mode="same"), dtype=np.float32)

class PointSpreadFunction:
    filt_inst = Filters()
    detection_thresholds = (200.0, 100.0, 50.0, 25.0, 15.0)
    min_epsf_stars = 20
    max_epsf_stars = 100
    max_direct_fit_stars = 50
    
    def __call__(self):
        pass
    
    @classmethod
    def extract(
        cls,
        image_data,
        header,
        galaxy_name,
        observatory,
        band,
        image_set,
        metadata_resolver=None,
        psfconv_config=None,
    ):
        target_name = f"{galaxy_name}/{observatory}/{band}"
        config = PSFConvolutionConfig.from_value(psfconv_config)
        measurement_image, measurement_header, measurement_info = cls.measurement_data(
            image_data,
            header,
            galaxy_name,
            observatory,
            band,
            metadata_resolver=metadata_resolver,
            psfconv_config=config,
        )

        measurement_start = perf_counter()
        method = None
        try:
            epsf_result = cls.get_epsf(
                measurement_image,
                measurement_header,
                galaxy_name,
                observatory,
                band,
                metadata_resolver=metadata_resolver,
                return_source=True,
                min_epsf_stars=config.min_epsf_stars,
            )
            if isinstance(epsf_result, tuple) and len(epsf_result) == 2:
                fwhm_val, method = epsf_result
            else:
                fwhm_val = epsf_result
                if cls._valid_fwhm(fwhm_val):
                    method = "empirical/predefined"
        except Exception as exc:
            warnings.warn(
                f"Empirical PSF extraction failed for {target_name}: {exc}. "
                "Trying safe fallbacks.",
                RuntimeWarning,
            )
            fwhm_val = None

        if not cls._valid_fwhm(fwhm_val):
            fwhm_val = cls.header_fwhm(measurement_header)
            if cls._valid_fwhm(fwhm_val):
                method = "header"
        if not cls._valid_fwhm(fwhm_val):
            fwhm_val = cls.observatory_median_fwhm(image_set, galaxy_name, observatory)
            if cls._valid_fwhm(fwhm_val):
                method = "observatory-median"
        if not cls._valid_fwhm(fwhm_val):
            try:
                fwhm_val = cls.measure_psf_fwhm(
                    measurement_image,
                    measurement_header,
                    threshold_sigma=15,
                )
                if cls._valid_fwhm(fwhm_val):
                    method = "direct-local" if measurement_info["region"] == "pretrim" else "direct-full"
            except Exception as exc:
                warnings.warn(
                    f"Direct PSF measurement failed for {target_name}: {exc}",
                    RuntimeWarning,
                )
                fwhm_val = None
        if not cls._valid_fwhm(fwhm_val):
            emit_alert(
                f"No valid PSF FWHM is available for {target_name}; "
                "storing NaN and skipping PSF convolution for this image.",
                context=target_name,
                dedupe_key="psf.fwhm.unavailable",
            )
            fwhm_val = np.nan
            method = "unavailable"
        else:
            fwhm_val = float(np.asarray(fwhm_val).item())

        image_set.psf = (galaxy_name, observatory, band, fwhm_val) # in ", fwhm * pixel_scale
        measurement_elapsed = perf_counter() - measurement_start
        fwhm_text = "NaN" if not cls._valid_fwhm(fwhm_val) else f"{fwhm_val:.6g}"
        fov_text = measurement_info["fov_arcsec"]
        emit_detail(
            f"PSF measurement {target_name}: region={measurement_info['region']}, "
            f"shape={measurement_info['original_shape']}->{measurement_info['measurement_shape']}, "
            f"fov(ny,nx)={fov_text} arcsec, "
            f"coverage={measurement_info['coverage']}:{measurement_info['coverage_fraction']:.3f}, "
            f"pixel_scale(ny,nx)={measurement_info['pixel_scale_arcsec']} arcsec/pixel, "
            f"prepare={measurement_info['prepare_seconds']:.3f}s, "
            f"method={method}, fwhm={fwhm_text} arcsec, measure={measurement_elapsed:.3f}s"
        )

    @classmethod
    def measurement_data(
        cls,
        image_data,
        header,
        galaxy_name,
        observatory,
        band,
        *,
        metadata_resolver=None,
        psfconv_config=None,
    ):
        """Return full data or an overlap-only native-WCS cutout for PSF measurement.

        This helper never updates ``image_set`` and never copies an error map.
        """
        config = PSFConvolutionConfig.from_value(psfconv_config)
        target_name = f"{galaxy_name}/{observatory}/{band}"
        original_shape = np.shape(image_data)
        if len(original_shape) != 2:
            raise ValueError(f"PSF measurement requires a 2-D image for {target_name}")

        use_pretrim = (
            config.psf_measurement_region == "pretrim"
            and config.pretrim_fov is not None
        )
        if not use_pretrim:
            try:
                matrix = cls._pixel_scale_matrix_arcsec(header)
                axis_scales = np.sqrt(np.sum(matrix ** 2, axis=0))[::-1]
                scale_text = tuple(round(float(value), 6) for value in axis_scales)
            except ValueError:
                scale_text = None
            return image_data, header, {
                "region": "full",
                "original_shape": original_shape,
                "measurement_shape": original_shape,
                "fov_arcsec": "full",
                "pixel_scale_arcsec": scale_text,
                "prepare_seconds": 0.0,
                "coverage": "full",
                "coverage_fraction": 1.0,
                "requested_shape": original_shape,
                "overlap_shape": original_shape,
            }

        center_coord = useful_functions.get_sky_loc(
            galaxy_name,
            header=header,
            metadata_resolver=metadata_resolver,
            required=True,
            image_shape=original_shape,
        )
        try:
            matrix = cls._pixel_scale_matrix_arcsec(header)
        except ValueError as exc:
            raise ValueError(f"Invalid WCS for PSF measurement cutout of {target_name}: {exc}") from exc
        axis_scales = np.sqrt(np.sum(matrix ** 2, axis=0))[::-1]
        scale_text = tuple(round(float(value), 6) for value in axis_scales)
        fov_text = tuple(
            round(float(value.to_value(u.arcsec)), 6)
            for value in config.pretrim_fov
        )

        from .registration import Register

        prepare_start = perf_counter()
        try:
            cutout_image, cutout_header, _, coverage_info = Register._trim_sky_with_info(
                image=image_data,
                header=header,
                error=None,
                skycoord=center_coord,
                size=config.pretrim_fov,
                mode="trim",
            )
        except NoOverlapError as exc:
            raise ValueError(
                f"PSF measurement FoV does not overlap the image for {target_name}: "
                f"requested(ny,nx)={fov_text} arcsec, input_shape={original_shape}, "
                f"pixel_scale(ny,nx)={scale_text} arcsec/pixel"
            ) from exc
        prepare_elapsed = perf_counter() - prepare_start

        if coverage_info["coverage"] == "partial":
            emit_alert(
                f"PSF measurement FoV is partially covered for {target_name}: "
                f"requested_shape={coverage_info['requested_shape']}, "
                f"overlap_shape={coverage_info['overlap_shape']}, "
                f"coverage={coverage_info['coverage_fraction']:.3f}; "
                "using only overlapping pixels.",
                context=target_name,
                dedupe_key="psf.measurement.partial_fov",
            )

        return cutout_image, cutout_header, {
            "region": "pretrim",
            "original_shape": original_shape,
            "measurement_shape": cutout_image.shape,
            "fov_arcsec": fov_text,
            "pixel_scale_arcsec": scale_text,
            "prepare_seconds": prepare_elapsed,
            **coverage_info,
        }

    @staticmethod
    def _valid_fwhm(value):
        try:
            value = float(np.asarray(value).item())
        except (TypeError, ValueError):
            return False
        return np.isfinite(value) and value > 0

    @classmethod
    def header_fwhm(cls, header):
        for key in ("SEEING", "HIERARCH SEEING"):
            value = header.get(key)
            if cls._valid_fwhm(value):
                return float(value)

        peeing = header.get("PEEING")
        pixel_scale = useful_functions.get_pixel_scale(header)
        if cls._valid_fwhm(peeing) and cls._valid_fwhm(pixel_scale):
            return float(peeing) * float(pixel_scale)
        return None

    @classmethod
    def observatory_median_fwhm(cls, image_set, galaxy_name, observatory):
        try:
            values = useful_functions.extract_values_recursive(
                image_set.psf[galaxy_name], observatory
            )
        except (KeyError, TypeError):
            return None
        valid = [float(value) for value in values if cls._valid_fwhm(value)]
        return float(np.nanmedian(valid)) if valid else None

    @classmethod
    def target_fwhm(cls, image_set, galaxy_name):
        """Return the largest valid PSF FWHM among this galaxy's images."""
        try:
            galaxy_psfs = image_set.psf[galaxy_name]
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"No PSF values are available for {galaxy_name}") from exc

        valid = []
        for observatory, bands in galaxy_psfs.items():
            if observatory == "max" or not isinstance(bands, dict):
                continue
            for value in bands.values():
                if cls._valid_fwhm(value):
                    valid.append(float(value))
        if not valid:
            cached_target = galaxy_psfs.get("max")
            if cls._valid_fwhm(cached_target):
                return float(cached_target)
            raise ValueError(f"No valid PSF FWHM values are available for {galaxy_name}")

        target = float(np.max(valid))
        galaxy_psfs["max"] = target
        return target

    @classmethod
    def extra_sigma_arcsec(cls, image_set, galaxy_name, observatory, band):
        try:
            input_fwhm = image_set.psf[galaxy_name][observatory][band]
        except (KeyError, TypeError, AttributeError):
            return None
        if not cls._valid_fwhm(input_fwhm):
            return None
        try:
            target_fwhm = cls.target_fwhm(image_set, galaxy_name)
        except ValueError:
            return None
        variance = (target_fwhm / 2.355) ** 2 - (float(input_fwhm) / 2.355) ** 2
        if variance <= np.finfo(float).eps:
            return 0.0
        return float(np.sqrt(variance))

    @staticmethod
    def _pixel_scale_matrix_arcsec(header):
        try:
            wcs = WCS(header).celestial
            matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float) * 3600.0
        except Exception as exc:
            raise ValueError(f"Could not determine celestial pixel scales: {exc}") from exc
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("Celestial WCS pixel scale matrix is invalid")
        metric = matrix.T @ matrix
        if not np.all(np.isfinite(metric)) or np.linalg.det(metric) <= 0:
            raise ValueError("Celestial WCS pixel scale matrix is singular")
        return matrix

    @classmethod
    def gaussian_matching_kernel(cls, header, sigma_arcsec, truncate=4.0):
        if not np.isfinite(sigma_arcsec) or sigma_arcsec <= 0:
            raise ValueError("Gaussian matching sigma must be finite and positive")
        matrix = cls._pixel_scale_matrix_arcsec(header)
        covariance = sigma_arcsec ** 2 * np.linalg.inv(matrix.T @ matrix)
        if not np.all(np.isfinite(covariance)):
            raise ValueError("Could not construct a finite PSF covariance in pixel coordinates")

        radius_x = max(1, int(np.ceil(truncate * np.sqrt(covariance[0, 0]))))
        radius_y = max(1, int(np.ceil(truncate * np.sqrt(covariance[1, 1]))))
        y, x = np.mgrid[-radius_y:radius_y + 1, -radius_x:radius_x + 1]
        inv_covariance = np.linalg.inv(covariance)
        exponent = (
            inv_covariance[0, 0] * x ** 2
            + 2.0 * inv_covariance[0, 1] * x * y
            + inv_covariance[1, 1] * y ** 2
        )
        kernel = np.exp(-0.5 * exponent)
        kernel_sum = np.sum(kernel, dtype=np.float64)
        if not np.isfinite(kernel_sum) or kernel_sum <= 0:
            raise ValueError("Gaussian matching kernel could not be normalized")
        return np.asarray(kernel / kernel_sum, dtype=np.float32)

    @staticmethod
    def _convolve_science(engine, image, kernel):
        finite = np.isfinite(image)
        clean = np.where(finite, image, 0.0).astype(np.float32, copy=False)
        convolved = engine.convolve(clean, kernel)
        if np.all(finite):
            return convolved, None

        support = engine.convolve(np.ones(image.shape, dtype=np.float32), kernel)
        valid_weight = engine.convolve(finite.astype(np.float32), kernel)
        factor = np.full(image.shape, np.nan, dtype=np.float32)
        minimum_weight = np.maximum(1e-7, 1e-6 * support)
        np.divide(support, valid_weight, out=factor, where=valid_weight > minimum_weight)
        return convolved * factor, factor

    @staticmethod
    def _convolve_std(engine, error, image, kernel, science_factor):
        error = np.asarray(error)
        finite_error = np.isfinite(error)
        if np.any(finite_error & (error < 0)):
            raise ValueError("STD error map contains finite negative values")

        science_valid = np.isfinite(image)
        error_valid = finite_error & science_valid
        variance = np.where(error_valid, error.astype(np.float64) ** 2, 0.0)
        convolved_variance = engine.convolve(variance.astype(np.float32), kernel ** 2)
        result = np.sqrt(np.maximum(convolved_variance, 0.0)).astype(np.float32)
        if science_factor is not None:
            result *= science_factor

        if not np.all(error_valid == science_valid):
            science_weight = engine.convolve(science_valid.astype(np.float32), kernel)
            error_weight = engine.convolve(error_valid.astype(np.float32), kernel)
            incomplete = error_weight < science_weight - np.maximum(1e-7, 1e-6 * science_weight)
            result[incomplete] = np.nan
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
        psfconv_config=None,
    ):
        """
        Convolve `image` with a Gaussian kernel of width `sigma_extra_pix` (pixels).
        If sigma_extra_pix==0, return original image.
        """
        from time import perf_counter

        target_name = f"{galaxy_name}/{observatory}/{band}"
        try:
            sig_i = image_set.psf[galaxy_name][observatory][band]
            sig_t = cls.target_fwhm(image_set, galaxy_name)
        except (KeyError, TypeError, ValueError):
            warnings.warn(
                f"Skipping PSF convolution for {target_name}: "
                "no valid PSF FWHM is available.",
                RuntimeWarning,
            )
            return
        if not cls._valid_fwhm(sig_i):
            warnings.warn(
                f"Skipping PSF convolution for {target_name}: no valid input PSF FWHM is available.",
                RuntimeWarning,
            )
            return

        sig_i = float(sig_i)
        variance_difference = (sig_t / 2.355) ** 2 - (sig_i / 2.355) ** 2
        if variance_difference <= np.finfo(float).eps:
            emit_detail(
                f"PSF convolution {target_name}: skipped (already at target {sig_t:.6g} arcsec)"
            )
            return
        sigma_arcsec = float(np.sqrt(variance_difference))
        config = PSFConvolutionConfig.from_value(psfconv_config)
        engine = convolution_engine or PSFConvolutionEngine(config)
        kernel = cls.gaussian_matching_kernel(
            header,
            sigma_arcsec,
            truncate=config.kernel_truncate,
        )

        image_data = np.asarray(image_data)
        if error_data is not None and np.shape(error_data) != image_data.shape:
            raise ValueError(
                f"STD error map shape {np.shape(error_data)} does not match science image "
                f"shape {image_data.shape} for {target_name}"
            )
        error_array = None if error_data is None else np.asarray(error_data)
        finite_error = None if error_array is None else np.isfinite(error_array)
        if error_array is not None:
            if np.any(finite_error & (error_array < 0)):
                raise ValueError(f"STD error map contains finite negative values for {target_name}")
        zero_error = (
            error_array is not None
            and np.any(finite_error)
            and np.all(error_array[finite_error] == 0)
        )

        science_start = perf_counter()
        convolved_img, science_factor = cls._convolve_science(
            engine,
            image_data,
            kernel,
        )
        science_elapsed = perf_counter() - science_start

        error_elapsed = 0.0
        error_status = "none"
        convolved_err = None
        if zero_error:
            error_status = "zero-skip"
        elif error_array is not None:
            error_start = perf_counter()
            convolved_err = cls._convolve_std(
                engine,
                error_array,
                image_data,
                kernel,
                science_factor,
            )
            error_elapsed = perf_counter() - error_start
            error_status = "convolved"

        image_set.update_data(convolved_img, galaxy_name, observatory, band)
        if convolved_err is not None:
            image_set.update_error(convolved_err, galaxy_name, observatory, band)

        emit_detail(
            f"PSF convolution {target_name}: backend={engine.backend}, "
            f"shape={image_data.shape}, kernel={kernel.shape}, "
            f"fwhm={sig_i:.6g}->{sig_t:.6g} arcsec, "
            f"science={science_elapsed:.3f}s, error={error_status}:{error_elapsed:.3f}s"
        )
    
    
    @classmethod
    def detect_star(
        cls,
        image,
        header,
        threshold_sigma,
        galaxy,
        observatory,
        band,
        metadata_resolver=None,
        min_epsf_stars=None,
    ):
        minimum_stars = cls.min_epsf_stars if min_epsf_stars is None else int(min_epsf_stars)
        try:
            curve = cls.filt_inst.ensure_filter(name=band, facility=observatory)
            mask = (curve.response != 0)
            wave = curve.wavelength[mask]
            min_wave = np.nanmin(wave)
        except (KeyError, ValueError):
            stars = cls.get_predefined_model(observatory, band)
            if stars is not None:
                return stars
            min_wave = np.nan
        
        if np.isfinite(min_wave) and min_wave > 12 * 1e4:
            return cls.get_predefined_model(observatory, band)

        gal_coord = useful_functions.get_sky_loc(
            galaxy,
            header=header,
            metadata_resolver=metadata_resolver,
            required=False,
            image_shape=image.shape,
        )
        if gal_coord is None:
            warnings.warn(
                f"Skipping empirical PSF extraction for {galaxy}/{observatory}/{band}: "
                "no validated galaxy coordinate lies inside the image.",
                RuntimeWarning,
            )
            return cls.get_predefined_model(observatory, band)

        resolver = metadata_resolver
        source_name = (
            resolver.get_coord_source(galaxy)
            if resolver is not None and hasattr(resolver, "get_coord_source")
            else "provided"
        )

        _, _, std = sigma_clipped_stats(image, sigma=10.0)
        if not np.isfinite(std) or std <= 0:
            warnings.warn(
                f"Skipping empirical PSF extraction for {galaxy}/{observatory}/{band}: "
                f"invalid image standard deviation {std}.",
                RuntimeWarning,
            )
            return cls.get_predefined_model(observatory, band)

        pixel_scale = useful_functions.get_pixel_scale(header)
        if not cls._valid_fwhm(pixel_scale):
            warnings.warn(
                f"Skipping empirical PSF extraction for {galaxy}/{observatory}/{band}: "
                "the FITS header has no valid pixel scale.",
                RuntimeWarning,
            )
            return cls.get_predefined_model(observatory, band)
        size = int(50 / pixel_scale) // 2 * 2 + 1
        size = max(7, size)
        hsize = (size - 1) / 2
        x_gal, y_gal = WCS(header).world_to_pixel(gal_coord)

        requested_threshold = float(threshold_sigma)
        thresholds = []
        for value in (requested_threshold, *cls.detection_thresholds[1:]):
            if value >= 10 and value not in thresholds:
                thresholds.append(value)

        for current_threshold in thresholds:
            try:
                sources = DAOStarFinder(
                    fwhm=3.0,
                    threshold=current_threshold * std,
                )(image)
            except Exception as exc:
                warnings.warn(
                    f"PSF source detection failed for {galaxy}/{observatory}/{band} "
                    f"at threshold {current_threshold:g}: {exc}",
                    RuntimeWarning,
                )
                continue
            if sources is None:
                emit_detail(
                    f"PSF candidates {galaxy}/{observatory}/{band}: coord={source_name}, "
                    f"threshold={current_threshold:g}, valid=0, selected=0"
                )
                continue

            x, y = sources["xcentroid"], sources["ycentroid"]
            valid_mask = (
                (x > hsize)
                & (x < (image.shape[1] - 1 - hsize))
                & (y > hsize)
                & (y < (image.shape[0] - 1 - hsize))
                & (np.sqrt((x - x_gal) ** 2 + (y - y_gal) ** 2) > (10 * size))
            )
            filtered_sources = sources[valid_mask]
            valid_count = len(filtered_sources)
            if valid_count < minimum_stars:
                emit_detail(
                    f"PSF candidates {galaxy}/{observatory}/{band}: coord={source_name}, "
                    f"threshold={current_threshold:g}, valid={valid_count}, selected=0"
                )
                continue

            selected_sources = cls._brightest_sources(
                filtered_sources,
                cls.max_epsf_stars,
            )
            stars_tbl = Table()
            stars_tbl["x"] = selected_sources["xcentroid"]
            stars_tbl["y"] = selected_sources["ycentroid"]
            try:
                stars = extract_stars(NDData(data=image), stars_tbl, size=size)
            except Exception as exc:
                warnings.warn(
                    f"Could not extract PSF stars for {galaxy}/{observatory}/{band} "
                    f"at threshold {current_threshold:g}: {exc}",
                    RuntimeWarning,
                )
                continue

            emit_detail(
                f"PSF candidates {galaxy}/{observatory}/{band}: coord={source_name}, "
                f"threshold={current_threshold:g}, valid={valid_count}, selected={len(stars)}"
            )
            if len(stars) >= minimum_stars:
                return stars

        target_name = f"{galaxy}/{observatory}/{band}"
        emit_alert(
            f"Insufficient empirical PSF stars for {target_name} "
            f"after {len(thresholds)} bounded detection attempts.",
            context=target_name,
            dedupe_key="psf.empirical.insufficient_stars",
        )
        return cls.get_predefined_model(observatory, band)

    @staticmethod
    def _brightest_sources(sources, limit):
        if len(sources) <= limit:
            return sources
        for column in ("flux", "peak"):
            if column in sources.colnames:
                values = np.asarray(sources[column], dtype=float)
                order = np.argsort(np.nan_to_num(values, nan=-np.inf))[::-1][:limit]
                return sources[order]
        return sources[:limit]


    @classmethod
    def get_epsf(
        cls,
        image=None,
        header=None,
        galaxy=None,
        observatory=None,
        band=None,
        metadata_resolver=None,
        return_source=False,
        min_epsf_stars=None,
    ):
        
        image = np.nan_to_num(image, nan=0.0)

        stars = cls.detect_star(image=image, header=header, threshold_sigma=200, 
                                galaxy=galaxy, observatory=observatory, band=band,
                                metadata_resolver=metadata_resolver,
                                min_epsf_stars=min_epsf_stars)
        
        psf = None
        psf_source = "empirical"
        if stars is not None and not isinstance(stars, str):
            try:
                epsf_builder = EPSFBuilder(oversampling=1, maxiters=10, progress_bar=False, smoothing_kernel="quartic", fitter=EPSFFitter(fit_boxsize=7))
                epsf, fitted_stars = epsf_builder(stars)
                psf = epsf.data
            except Exception as exc:
                warnings.warn(
                    f"Empirical PSF construction failed for {galaxy}/{observatory}/{band}: {exc}",
                    RuntimeWarning,
                )
                stars = cls.get_predefined_model(observatory, band)

        if psf is None:
            if stars is None:
                return (None, None) if return_source else None
            psf_source = "predefined"
            try:
                psf = fits.getdata(stars, ext=1)
                header = fits.getheader(stars, ext=1)
            except (IndexError, KeyError, OSError, ValueError):
                try:
                    psf = fits.getdata(stars)
                    header = fits.getheader(stars)
                except (IndexError, KeyError, OSError, ValueError) as exc:
                    warnings.warn(
                        f"Could not load predefined PSF for {galaxy}/{observatory}/{band}: {exc}",
                        RuntimeWarning,
                    )
                    return (None, None) if return_source else None
        
        psf_center = (psf.shape[0]/2, psf.shape[1]/2)
        if psf_center[0] > 200:
            new_size = int(psf.shape[0] * 0.15)
            if new_size % 2 == 0:
                new_size += 1
            
            half_size = new_size // 2
            cy, cx = int(psf_center[0]), int(psf_center[1])
            
            psf = psf[cy - half_size : cy + half_size + 1, 
                      cx - half_size : cx + half_size + 1]
            psf_center = (psf.shape[0]/2, psf.shape[1]/2)
            
        try:
            fit_2d = fit_2dgaussian(psf, xypos=psf_center, fix_fwhm=False)
            fwhm = fit_2d.results["fwhm_fit"].value
        except Exception as exc:
            warnings.warn(
                f"PSF profile fit failed for {galaxy}/{observatory}/{band}: {exc}",
                RuntimeWarning,
            )
            return (None, None) if return_source else None
        
        pixel_scale = useful_functions.get_pixel_scale(header)
        
        result = fwhm * pixel_scale
        return (result, psf_source) if return_source else result
    
    
    @classmethod
    def get_predefined_model(cls, observatory, band):
        filter_dir = resources.files("Spec7DT.reference.psfs")
        for filepath in filter_dir.iterdir():
            if filepath.name.endswith('.fits'):
                try:
                    with resources.as_file(filepath) as file_path:
                        file_path = str(file_path)
                        obs_file = Parsers._observatory_name_parser(file_path)
                        band_file = Parsers._band_name_parser(file_path, cls.filt_inst)
                        if (obs_file.lower() == observatory.lower()) & (band_file.lower() == band.lower()):
                            emit_detail("Load Pre-defined PSF model.")
                            return file_path
                        else:
                            continue
                except:
                    continue
        

    @staticmethod
    def measure_fwhm_gaussian(image, x_center, y_center, box_size=21):
        """
        Measure FWHM by fitting 2D Gaussian to PSF
        """
        # Extract cutout around star
        y, x = np.ogrid[:box_size, :box_size]
        x_start = int(x_center - box_size//2)
        y_start = int(y_center - box_size//2)
        
        cutout = image[y_start:y_start+box_size, x_start:x_start+box_size]
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[:box_size, :box_size]
        
        # Initial parameter guess
        amplitude = np.max(cutout)
        x_mean = box_size // 2
        y_mean = box_size // 2
        
        # Fit 2D Gaussian
        g_init = models.Gaussian2D(amplitude=amplitude, 
                                x_mean=x_mean, y_mean=y_mean,
                                x_stddev=box_size * 0.05, y_stddev=box_size * 0.05)
        fit_g = fitting.TRFLSQFitter()
        g = fit_g(g_init, x_grid, y_grid, cutout)
        
        # Convert stddev to FWHM
        fwhm_x = 2.355 * g.x_stddev.value
        fwhm_y = 2.355 * g.y_stddev.value
        fwhm_avg = (fwhm_x + fwhm_y) / 2
        
        return fwhm_avg

    @classmethod
    def measure_psf_fwhm(cls, image, header, threshold_sigma=15):
        """
        Complete pipeline: detect stars and measure FWHM
        """
        height, width = image.shape
        pixel_scale = useful_functions.get_pixel_scale(header)
        if not cls._valid_fwhm(pixel_scale):
            warnings.warn("Cannot measure PSF FWHM: FITS header has no valid pixel scale.", RuntimeWarning)
            return None

        box_size = int(np.ceil(10.0 / pixel_scale))
        box_size = min(51, max(15, box_size))
        if box_size % 2 == 0:
            box_size = box_size + 1 if box_size < 51 else box_size - 1

        _, _, std = sigma_clipped_stats(image, sigma=10.0)
        if not np.isfinite(std) or std <= 0:
            return None

        try:
            daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma * std)
            sources = daofind(image)
        except Exception as exc:
            warnings.warn(f"Direct PSF source detection failed: {exc}", RuntimeWarning)
            return None
        if sources is None:
            return None

        half_size = box_size // 2
        x = np.asarray(sources["xcentroid"], dtype=float)
        y = np.asarray(sources["ycentroid"], dtype=float)
        interior = (
            (x >= half_size)
            & (x < width - half_size)
            & (y >= half_size)
            & (y < height - half_size)
        )
        sources = cls._brightest_sources(sources[interior], cls.max_direct_fit_stars)

        fwhm_measurements = []
        for source in sources:
            x, y = source['xcentroid'], source['ycentroid']
            try:
                fwhm = cls.measure_fwhm_gaussian(image, x, y, box_size=box_size)
            except (ValueError, RuntimeError, TypeError, IndexError):
                continue
            if np.isfinite(fwhm) and 0 < fwhm < box_size / 2:
                fwhm_measurements.append(float(fwhm))

        if not fwhm_measurements:
            return None
        return float(np.nanmedian(fwhm_measurements)) * float(pixel_scale)
