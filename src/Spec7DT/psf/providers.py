from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate

from .models import PixelSampling, PSFEnsemble, PSFRequest, PSFSample, ResolvedPSF
from .render import (
    representative_from_ensemble,
    singleton_ensemble,
    summarize_spatial_ensemble,
)
from .products import sampling_from_header


class PSFProviderError(RuntimeError):
    pass


class ProviderUnavailable(PSFProviderError):
    pass


class ProviderRejected(PSFProviderError):
    pass


_PROVIDER_FACTORIES: list[Callable] = []


def register_provider_factory(factory):
    if factory not in _PROVIDER_FACTORIES:
        _PROVIDER_FACTORIES.append(factory)
    return factory


def registered_provider_factories():
    return tuple(_PROVIDER_FACTORIES)


class PSFProviderRegistry:
    def __init__(self, providers=()):
        self.providers = list(providers)

    @classmethod
    def with_registered(cls, *, validator):
        providers = [factory(validator=validator) for factory in registered_provider_factories()]
        return cls(providers)

    def register(self, provider):
        self.providers.append(provider)

    def resolve(self, request, config):
        ranked = []
        for provider in self.providers:
            score = int(provider.match(request, config))
            if score > 0:
                ranked.append((score, provider))
        diagnostics = []
        for score in sorted({item[0] for item in ranked}, reverse=True):
            group = [provider for item_score, provider in ranked if item_score == score]
            if len(group) > 1:
                names = ", ".join(sorted(provider.name for provider in group))
                raise ProviderRejected(f"Ambiguous PSF providers at priority {score}: {names}")
            provider = group[0]
            try:
                resolved = provider.resolve(request, config)
            except ProviderUnavailable as exc:
                diagnostics.append(f"{provider.name}: {exc}")
                continue
            if resolved is None:
                diagnostics.append(f"{provider.name}: unavailable")
                continue
            resolved.diagnostics[:0] = diagnostics
            return resolved
        raise ProviderUnavailable("; ".join(diagnostics) or "no PSF provider matched the product")


@dataclass
class FunctionalPSFProvider:
    name: str
    priority: int
    matcher: Callable
    resolver: Callable
    terminal_on_error: bool = False

    def match(self, request, config):
        return self.priority if self.matcher(request, config) else 0

    def resolve(self, request, config):
        try:
            return self.resolver(request, config)
        except (ProviderUnavailable, ProviderRejected):
            raise
        except Exception as exc:
            if self.terminal_on_error:
                raise ProviderRejected(str(exc)) from exc
            raise ProviderUnavailable(str(exc)) from exc


class ExternalPSFProvider:
    name = "external-manifest"
    priority = 300
    _manifest_cache = {}
    _model_cache = {}

    def __init__(self, *, validator):
        self.validator = validator

    @classmethod
    def _manifest(cls, value):
        if value is None:
            return [], None, 1
        if isinstance(value, Mapping):
            raw = dict(value)
            base = None
        else:
            path = Path(value).expanduser().resolve()
            stamp = (str(path), path.stat().st_mtime_ns)
            if stamp in cls._manifest_cache:
                return cls._manifest_cache[stamp]
            with path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
            base = path.parent
        version = int(raw.get("schema_version", 1)) if isinstance(raw, Mapping) else 1
        entries = raw.get("models", raw)
        normalized = []
        if isinstance(entries, Mapping):
            for key, value_item in entries.items():
                item = {"path": value_item} if isinstance(value_item, (str, Path)) else dict(value_item)
                if "." in str(key):
                    facility, channel = str(key).split(".", 1)
                    item.setdefault("observatory", facility)
                    item.setdefault("band", channel)
                normalized.append(item)
        else:
            normalized = [dict(item) for item in entries]
        result = (normalized, base, version)
        if not isinstance(value, Mapping):
            cls._manifest_cache[stamp] = result
        return result

    @staticmethod
    def _selector(entry):
        selector = dict(entry.get("selector", {}))
        selector.setdefault("facility", entry.get("observatory"))
        selector.setdefault("channel", entry.get("band"))
        return selector

    @classmethod
    def _entry_matches(cls, entry, request):
        selector = cls._selector(entry)
        checks = (
            ("product_id", request.product.product_id),
            ("facility", request.product.facility or request.observatory),
            ("channel", request.product.channel or request.band),
        )
        for key, actual in checks:
            expected = selector.get(key)
            if expected is not None and str(expected).lower() != str(actual).lower():
                return False
        region = selector.get("valid_region")
        if region is not None and request.tile_bounds is not None:
            x0, y0, x1, y1 = map(float, region)
            tx0, ty0, tx1, ty1 = request.tile_bounds
            if tx0 < x0 or ty0 < y0 or tx1 > x1 or ty1 > y1:
                return False
        return any(cls._selector(entry).get(key) is not None for key in ("product_id", "facility", "channel"))

    def _matches(self, request, config):
        entries, _, _ = self._manifest(config.psf_manifest)
        matches = [entry for entry in entries if self._entry_matches(entry, request)]
        if not matches:
            return []
        def specificity(entry):
            selector = self._selector(entry)
            return (
                100 if selector.get("product_id") is not None else 0
            ) + (
                10 if selector.get("valid_region") is not None else 0
            ) + sum(selector.get(key) is not None for key in ("facility", "channel"))
        maximum = max(specificity(entry) for entry in matches)
        return [entry for entry in matches if specificity(entry) == maximum]

    def match(self, request, config):
        matches = self._matches(request, config)
        if not matches:
            return 0
        exact = any(self._selector(item).get("product_id") is not None for item in matches)
        return self.priority + (10 if exact else 0)

    @staticmethod
    def _path(value, base):
        path = Path(value).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        if not path.is_file():
            raise ProviderRejected(f"External PSF does not exist: {path}")
        return path

    def _load_one(self, spec, base, config, source_suffix=""):
        representation = dict(spec.get("representation", {}))
        merged = dict(representation)
        merged.update(spec)
        path = self._path(merged["path"], base)
        hdu = merged.get("hdu", 0)
        cache_key = (
            str(path.resolve()),
            path.stat().st_mtime_ns,
            str(hdu),
            repr(merged.get("sampling")),
            merged.get("pixel_scale_arcsec"),
            merged.get("frame"),
            float(merged.get("position_angle_deg", 0.0)),
            merged.get("pixel_integrated"),
            repr(self._selector(spec).get("valid_region")),
            repr(spec.get("provenance", {})),
            float(config.psf_edge_mass_tolerance),
            float(config.negative_peak_tolerance),
            float(config.negative_mass_tolerance),
        )
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached
        with fits.open(path, memmap=False) as hdul:
            data = np.asarray(hdul[hdu].data, dtype=float)
            header = hdul[hdu].header
        sampling_value = merged.get("sampling")
        if sampling_value is None:
            scale = merged.get("pixel_scale_arcsec")
            if scale is None:
                scale = header.get("PSCALE") or header.get("PIXSCALE")
            if scale is None:
                inferred = sampling_from_header(header)
                if inferred is None:
                    raise ProviderRejected(f"External PSF sampling is missing: {path}")
                sampling = inferred
            else:
                sampling = PixelSampling.isotropic(float(scale), frame=str(merged.get("frame", "image")))
        else:
            sampling = PixelSampling.from_value(sampling_value)
        angle = float(merged.get("position_angle_deg", 0.0))
        if angle:
            data = rotate(data, angle, reshape=True, order=1, mode="constant", cval=0.0, prefilter=False)
            data = np.maximum(data, 0.0)
            if data.shape[0] % 2 == 0:
                data = data[:-1]
            if data.shape[1] % 2 == 0:
                data = data[:, :-1]
        model = self.validator(
            data,
            sampling,
            f"external:{path.name}{source_suffix}",
            edge_tolerance=config.psf_edge_mass_tolerance,
            negative_peak_tolerance=config.negative_peak_tolerance,
            negative_mass_tolerance=config.negative_mass_tolerance,
            frame=sampling.frame,
            provenance={
                "path": str(path),
                "hdu": hdu,
                "pixel_integrated": merged.get("pixel_integrated"),
                **dict(spec.get("provenance", {})),
            },
        )
        model.validity_region = self._selector(spec).get("valid_region")
        if len(self._model_cache) >= 512:
            self._model_cache.pop(next(iter(self._model_cache)))
        self._model_cache[cache_key] = model
        return model

    def resolve(self, request, config):
        entries, base, version = self._manifest(config.psf_manifest)
        matches = self._matches(request, config)
        if not matches:
            raise ProviderUnavailable("no manifest entry")
        if len(matches) > 1:
            raise ProviderRejected(
                f"External PSF selector is ambiguous for {request.galaxy}/{request.observatory}/{request.band}"
            )
        entry = matches[0]
        representation = dict(entry.get("representation", {}))
        validation = dict(entry.get("spatial_validation", {}))
        applicability = entry.get("applicability")
        if applicability is None and entry.get("kind") == "effective_mosaic":
            applicability = "output_product"
        if request.product.requires_output_effective_psf and applicability != "output_product":
            raise ProviderRejected(
                "Coadd PSF manifest entry must declare applicability='output_product'"
            )
        sample_specs = validation.get("samples") or representation.get("samples") or entry.get("samples")
        samples = []
        if sample_specs:
            for index, sample_spec in enumerate(sample_specs):
                spec = dict(entry)
                spec.update(dict(sample_spec))
                model = self._load_one(spec, base, config, source_suffix=f":sample{index}")
                position = sample_spec.get("position")
                samples.append(PSFSample(
                    model,
                    None if position is None else tuple(map(float, position)),
                    weight=float(sample_spec.get("weight", 1.0)),
                    metadata={"sample_index": index},
                ))
        else:
            model = self._load_one(entry, base, config)
            samples.append(PSFSample(model, position=None))

        ensemble = PSFEnsemble(samples, provenance={
            "manifest_schema": version,
            "legacy_selector": version == 1 and self._selector(entry).get("product_id") is None,
            "resolved_product_id": request.product.product_id,
            "applicability": applicability,
            **dict(entry.get("provenance", {})),
        })
        if request.product.requires_output_effective_psf:
            if len({sample.position for sample in samples}) < 5:
                declared = validation.get("metrics")
                if not config.allow_unvalidated_stationary and not declared:
                    raise ProviderRejected(
                        "Coadd PSF requires center/corner spatial validation samples or declared validation metrics"
                    )
                if declared:
                    ensemble.spatial_metrics = {str(key): float(value) for key, value in declared.items()}
                    aliases = {
                        "spatial_r80_spread": config.spatial_r80_tolerance,
                        "spatial_ee_max": config.spatial_ee_tolerance,
                        "spatial_l1_max": config.spatial_l1_tolerance,
                        "spatial_centroid_max": config.kernel_centroid_tolerance,
                    }
                    missing_metrics = sorted(set(aliases) - set(ensemble.spatial_metrics))
                    if missing_metrics:
                        raise ProviderRejected(
                            "Declared coadd PSF validation is missing: " + ", ".join(missing_metrics)
                        )
                    exceeded = [
                        f"{key}={ensemble.spatial_metrics[key]:.4g} > {limit:.4g}"
                        for key, limit in aliases.items()
                        if key in ensemble.spatial_metrics and ensemble.spatial_metrics[key] > limit
                    ]
                    if exceeded:
                        raise ProviderRejected(
                            "Declared coadd PSF spatial validation failed: " + "; ".join(exceeded)
                        )
            else:
                ensemble.spatial_metrics = summarize_spatial_ensemble(
                    ensemble,
                    r80_tolerance=config.spatial_r80_tolerance,
                    ee_tolerance=config.spatial_ee_tolerance,
                    l1_tolerance=config.spatial_l1_tolerance,
                    centroid_tolerance=config.kernel_centroid_tolerance,
                )
        representative = representative_from_ensemble(
            ensemble,
            "external-ensemble",
            validator=self.validator,
            edge_tolerance=config.psf_edge_mass_tolerance,
        )
        return ResolvedPSF(representative, ensemble, self.name)

