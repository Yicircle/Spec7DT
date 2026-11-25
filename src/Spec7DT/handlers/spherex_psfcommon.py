"""Analytic PSF provider for PSF-homogenized SPHEREx mosaic cubes.

The producer of ``PSFMODE='CUBE-COMMON'`` cubes matched every plane to one
circular Gaussian of sigma ``TARGET_SIGMA`` native pixels, defined before pixel
integration.  A native pixel therefore sees ``Gaussian(sigma) * box(1 px)``.
This provider renders that pixel response function, validates it against the
producer's model-achieved ``PSF_GRID`` tiles, and never synthesizes a PSF from
the pre-homogenization exposures.  All SPHEREx-specific rules live here; the
safe matcher consumes only the generic ``ResolvedPSF``.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..psf.models import PixelSampling, ResolvedPSF
from ..psf.providers import (
    ExternalPSFProvider,
    ProviderRejected,
    ProviderUnavailable,
    register_provider_factory,
)
from ..psf.render import (
    centered_coordinates,
    common_shape,
    encircled_energy_difference,
    psf_statistics,
    singleton_ensemble,
)


OVERSAMPLE = 5


def spherex_prf(sigma_px, oversample=OVERSAMPLE):
    # Imported lazily: handlers must not import utils at module import time.
    from ..utils.spherex_cube import spherex_prf as render

    return render(sigma_px, oversample)


def _is_homogenized(product):
    return (
        str(product.facility or "").lower() == "spherex"
        and product.product_kind == "coadd"
        and bool(dict(product.provenance or {}).get("psf_homogenized"))
    )


def _r80_pixels(sigma_px):
    """r80 of the oversampled PRF in native pixels (continuous estimate)."""

    prf = spherex_prf(sigma_px, OVERSAMPLE)
    y, x = centered_coordinates(prf.shape)
    radius = np.hypot(x, y).ravel() / OVERSAMPLE
    order = np.argsort(radius)
    cumulative = np.cumsum(prf.ravel()[order])
    return float(np.interp(0.8, cumulative, radius[order]))


def broadened_sigma(sigma_px, r80_excess, *, tolerance=1e-6, max_iterations=100):
    """Return sigma' with r80(PRF sigma') = (1 + excess) * r80(PRF sigma)."""

    sigma_px = float(sigma_px)
    excess = float(r80_excess)
    if not np.isfinite(excess) or excess <= 0:
        return sigma_px
    goal = (1.0 + excess) * _r80_pixels(sigma_px)
    low, high = sigma_px, sigma_px * (1.0 + excess)
    while _r80_pixels(high) < goal:
        high *= 1.5
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        if _r80_pixels(middle) < goal:
            low = middle
        else:
            high = middle
        if high - low <= tolerance * sigma_px:
            break
    return 0.5 * (low + high)


class SpherexPSFCommonProvider:
    name = "spherex-psfcommon-gaussian"
    # Higher than ExternalPSFProvider (300/310), so a stale synthesized
    # manifest entry can never win over the homogenized cube's own PSF.
    priority = 320
    _grid_cache = {}
    _resolved_cache = {}

    def __init__(self, *, validator):
        self.validator = validator

    def match(self, request, config):
        return self.priority if _is_homogenized(request.product) else 0

    @classmethod
    def _psf_grid(cls, path, grid_hdu, plane):
        path = Path(path).expanduser().resolve()
        key = (str(path), path.stat().st_mtime_ns, str(grid_hdu), int(plane))
        cached = cls._grid_cache.get(key)
        if cached is not None:
            return cached
        with fits.open(path, memmap=True) as hdul:
            if grid_hdu not in hdul:
                raise ProviderRejected(f"SPHEREx cube {path.name} has no {grid_hdu} HDU")
            grid = np.array(hdul[grid_hdu].section[int(plane)], dtype=float, copy=True)
        if grid.ndim != 3:
            raise ProviderRejected(
                f"SPHEREx {grid_hdu} plane {plane} must be (tiles, ny, nx); got {grid.shape}"
            )
        if len(cls._grid_cache) >= 256:
            cls._grid_cache.pop(next(iter(cls._grid_cache)))
        cls._grid_cache[key] = grid
        return grid

    @staticmethod
    def spatial_metrics(grid, native_prf):
        """Compare every usable PSF_GRID tile with the native-sampled PRF."""

        reference = np.asarray(native_prf, dtype=float)
        reference = reference / np.sum(reference)
        _, _, reference_r80 = psf_statistics(reference, 1.0)
        l1_max = ee_max = centroid_max = 0.0
        r80_values = [reference_r80]
        used = 0
        for tile in grid:
            tile = np.asarray(tile, dtype=float)
            if not np.all(np.isfinite(tile)) or np.sum(tile) <= 0:
                continue
            # Tiny negative model values (~1e-7) are numerical noise.
            tile = np.maximum(tile, 0.0)
            a, b = common_shape((tile / np.sum(tile), reference))
            a /= np.sum(a)
            b /= np.sum(b)
            l1_max = max(l1_max, float(np.sum(np.abs(a - b))))
            ee_max = max(ee_max, encircled_energy_difference(a, b))
            y, x = centered_coordinates(a.shape)
            centroid_max = max(
                centroid_max,
                float(np.hypot(np.sum(x * a), np.sum(y * a))),
            )
            r80_values.append(psf_statistics(a, 1.0)[2])
            used += 1
        r80_values = np.asarray(r80_values, dtype=float)
        spread = float((np.max(r80_values) - np.min(r80_values)) / np.mean(r80_values))
        return {
            "spatial_l1_max": l1_max,
            "spatial_ee_max": ee_max,
            "spatial_centroid_max": centroid_max,
            "spatial_r80_spread": spread,
        }, used

    @staticmethod
    def _check_tolerances(metrics, config, label):
        limits = {
            "spatial_l1_max": config.spatial_l1_tolerance,
            "spatial_ee_max": config.spatial_ee_tolerance,
            "spatial_centroid_max": config.kernel_centroid_tolerance,
            "spatial_r80_spread": config.spatial_r80_tolerance,
        }
        failures = [
            f"{name}={metrics[name]:.4g} > {limit:.4g}"
            for name, limit in limits.items()
            if metrics[name] > limit
        ]
        if failures:
            raise ProviderRejected(
                f"{label}: analytic PRF does not match PSF_GRID: " + "; ".join(failures)
            )

    def _warn_shadowed_manifest(self, request, config):
        if config.psf_manifest is None:
            return
        try:
            external = ExternalPSFProvider(validator=self.validator)
            if not external.match(request, config):
                return
            names = []
            for entry in external._matches(request, config):
                names.append(str(entry.get("path") or entry.get("product_id") or entry))
        except Exception:
            return
        from ..utils.reporting import emit_alert

        emit_alert(
            "PSF manifest entries also match this PSF-homogenized SPHEREx product and "
            f"are ignored in favour of {self.name}: {', '.join(names) or 'unnamed entry'}",
            context=f"{request.galaxy}/{request.observatory}/{request.band}",
            dedupe_key=("spherex-psfcommon-shadowed", request.product.product_id),
        )

    def resolve(self, request, config):
        product = request.product
        provenance = dict(product.provenance or {})
        label = f"{request.galaxy}/{request.observatory}/{request.band}"
        sampling = product.sampling
        if sampling is None:
            raise ProviderRejected(f"{label}: SPHEREx coadd has no valid angular sampling")
        try:
            target_sigma = float(provenance["target_sigma_pixel"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderRejected(f"{label}: homogenized product has no target sigma") from exc
        if not np.isfinite(target_sigma) or target_sigma <= 0:
            raise ProviderRejected(f"{label}: invalid target sigma {target_sigma}")
        plane = provenance.get("source_plane", product.plane)
        if plane is None or product.source_path is None:
            raise ProviderUnavailable(f"{label}: source cube plane is unknown")
        plane = int(plane)
        policy = str(provenance.get("psf_mismatch_policy", "keep"))
        quality = dict(provenance.get("psf_quality") or {})
        psfq = str(quality.get("psfq") or "").upper()
        r80_excess = quality.get("r80_excess")
        grid_hdu = provenance.get("psf_grid_hdu") or "PSF_GRID"

        self._warn_shadowed_manifest(request, config)

        sigma = target_sigma
        if policy == "broaden" and psfq == "MISMATCH" and r80_excess is not None:
            sigma = broadened_sigma(target_sigma, float(r80_excess))

        source_path = Path(product.source_path).expanduser()
        cache_payload = (
            str(source_path.resolve()),
            source_path.stat().st_mtime_ns,
            plane,
            target_sigma,
            sigma,
            policy,
            sampling.matrix_arcsec,
            sampling.frame,
            float(config.psf_edge_mass_tolerance),
            float(config.negative_peak_tolerance),
            float(config.negative_mass_tolerance),
            float(config.spatial_l1_tolerance),
            float(config.spatial_ee_tolerance),
            float(config.spatial_r80_tolerance),
            float(config.kernel_centroid_tolerance),
        )
        cache_key = sha256(repr(cache_payload).encode("utf-8")).hexdigest()
        cached = self._resolved_cache.get(cache_key)
        if cached is not None:
            return ResolvedPSF(cached[0], cached[1], self.name)

        # The PSF_GRID tiles describe the achieved target PSF, so validate
        # them against the target sigma before any optional broadening.
        grid = self._psf_grid(source_path, grid_hdu, plane)
        metrics, tiles_used = self.spatial_metrics(grid, spherex_prf(target_sigma, 1))
        if tiles_used == 0:
            raise ProviderRejected(f"{label}: {grid_hdu} plane {plane} has no usable tiles")
        self._check_tolerances(metrics, config, label)

        oversampled = PixelSampling(
            np.asarray(sampling.matrix_arcsec, dtype=float) / OVERSAMPLE,
            frame=sampling.frame,
        )
        model = self.validator(
            spherex_prf(sigma, OVERSAMPLE),
            oversampled,
            f"spherex-psfcommon:sigma={sigma:.6g}",
            edge_tolerance=config.psf_edge_mass_tolerance,
            negative_peak_tolerance=config.negative_peak_tolerance,
            negative_mass_tolerance=config.negative_mass_tolerance,
            frame="image",
            provenance={
                "provider": self.name,
                "plane": plane,
                "target_sigma_pixel": target_sigma,
                "sigma_pixel": sigma,
                "broadened": sigma != target_sigma,
                "psfq": psfq or None,
                "r80_excess": r80_excess,
                "psf_mismatch_policy": policy,
                "oversample": OVERSAMPLE,
                "pixel_integrated": True,
            },
        )
        model.quality.update(metrics)
        model.quality["psf_grid_tiles"] = tiles_used
        ensemble = singleton_ensemble(model)
        ensemble.spatial_metrics = dict(metrics)
        ensemble.provenance = {
            "provider": self.name,
            "psf_grid_hdu": grid_hdu,
            "psf_grid_tiles": tiles_used,
            "target_sigma_pixel": target_sigma,
            "sigma_pixel": sigma,
        }
        if len(self._resolved_cache) >= 512:
            self._resolved_cache.pop(next(iter(self._resolved_cache)))
        self._resolved_cache[cache_key] = (model, ensemble)
        return ResolvedPSF(model, ensemble, self.name)


@register_provider_factory
def _spherex_psfcommon_provider_factory(*, validator):
    return SpherexPSFCommonProvider(validator=validator)


__all__ = ["SpherexPSFCommonProvider", "broadened_sigma"]
