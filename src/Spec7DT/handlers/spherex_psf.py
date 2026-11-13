"""SPHEREx product adapter and official Level-2 ePSF provider.

Mission-specific version, detector-zone, and cutout-coordinate rules are
intentionally isolated here.  The safe matcher consumes only generic product
metadata and PSF ensembles.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
from astropy.io import fits

from ..psf.models import (
    ImageProductMetadata,
    PixelSampling,
    PSFEnsemble,
    PSFSample,
    ResolvedPSF,
)
from ..psf.products import (
    register_product_adapter,
    sampling_from_header,
    wcs_metadata_from_header,
)
from ..psf.providers import ProviderRejected, ProviderUnavailable, register_provider_factory
from ..psf.render import (
    render_oversampled_psf,
    representative_from_ensemble,
    summarize_spatial_ensemble,
)


def _version_major(value):
    match = re.search(r"(?:^|[^0-9])(\d+)(?:\.(\d+))?", str(value or ""))
    return None if match is None else int(match.group(1))


def _is_spherex_header(header):
    values = (
        header.get("TELESCOP"),
        header.get("OBSERVAT"),
        header.get("MISSION"),
        header.get("INSTRUME"),
    )
    return any("spherex" in str(value).lower() for value in values if value is not None)


class SpherexProductAdapter:
    priority = 100

    @staticmethod
    def probe(hdul, filepath, image_index, merged_header):
        extension_names = {str(hdu.header.get("EXTNAME", "")).strip().upper() for hdu in hdul}
        has_psf_extension = "EPSF" in extension_names or "PSF" in extension_names
        filename = Path(filepath).name.lower()
        filename_hint = "spherex" in filename or "_spx_" in filename
        if not _is_spherex_header(merged_header) and not (
            filename_hint
            and has_psf_extension
            and merged_header.get("DETECTOR") is not None
        ):
            return None
        version = merged_header.get("VERSION")
        cutout = "CRPIX1A" in merged_header and "CRPIX2A" in merged_header
        embedded = None
        if "EPSF" in extension_names:
            embedded = {"type": "oversampled_table", "hdu": "EPSF", "usable": True}
        elif "PSF" in extension_names:
            embedded = {"type": "legacy_psf_cube", "hdu": "PSF", "usable": False}
        major = _version_major(version)
        return ImageProductMetadata(
            product_kind="cutout" if cutout else "exposure",
            facility="SPHEREx",
            instrument="SPHEREx-LVF",
            release="qr3" if major is not None and major >= 7 else "qr2",
            version=None if version is None else str(version),
            detector=merged_header.get("DETECTOR"),
            source_path=str(filepath),
            hdu=hdul[image_index].header.get("EXTNAME", image_index),
            sampling=sampling_from_header(merged_header),
            output_wcs=wcs_metadata_from_header(merged_header),
            requires_output_effective_psf=False,
            embedded_psf=embedded,
            provenance={"adapter": "spherex-level2", "image_hdu_index": int(image_index)},
        )


register_product_adapter(SpherexProductAdapter())


class SpherexEPSFProvider:
    name = "spherex-level2-epsf"
    priority = 200
    _table_cache = {}
    _registry_cache = {}
    _checksum_cache = {}
    _ensemble_cache = {}

    def __init__(self, *, validator):
        self.validator = validator

    @staticmethod
    def _is_level2(product):
        return (
            str(product.facility or "").lower() == "spherex"
            and product.product_kind in {"exposure", "cutout"}
        )

    def match(self, request, config):
        return self.priority if self._is_level2(request.product) else 0

    @classmethod
    def _registry(cls, value):
        if value is None:
            return [], None
        if isinstance(value, Mapping):
            raw = dict(value)
            base = None
        else:
            path = Path(value).expanduser().resolve()
            stamp = (str(path), path.stat().st_mtime_ns)
            if stamp in cls._registry_cache:
                return cls._registry_cache[stamp]
            with path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
            base = path.parent
        if int(raw.get("schema_version", 1)) != 1:
            raise ProviderRejected("Unsupported calibration registry schema")
        entries = [dict(item) for item in raw.get("calibrations", ())]
        result = (entries, base)
        if not isinstance(value, Mapping):
            cls._registry_cache[stamp] = result
        return result

    @staticmethod
    def _verify_checksum(path, expected):
        if not expected:
            return
        path = Path(path)
        key = (str(path.resolve()), path.stat().st_mtime_ns, str(expected).lower())
        if SpherexEPSFProvider._checksum_cache.get(key):
            return
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != str(expected).lower():
            raise ProviderRejected(f"Calibration checksum mismatch: {path}")
        SpherexEPSFProvider._checksum_cache[key] = True

    @classmethod
    def _calibration_path(cls, product, config):
        embedded = product.embedded_psf or {}
        major = _version_major(product.version)
        if embedded.get("type") == "oversampled_table" and (major is None or major >= 7):
            path = Path(product.source_path).expanduser()
            if not path.is_file():
                raise ProviderRejected(f"Embedded ePSF source file is missing: {path}")
            return path, embedded.get("hdu", "EPSF"), None, "embedded"

        entries, base = cls._registry(config.calibration_registry)
        detector = str(product.detector)

        def is_supported_release(item):
            release = str(item.get("release", "")).strip().lower()
            if release.startswith(("qr3", "dr1")):
                return True
            major = _version_major(release)
            return major is not None and major >= 7

        matches = [
            item for item in entries
            if str(item.get("adapter", "spherex_l2_epsf")).lower() == "spherex_l2_epsf"
            and str(item.get("detector")) == detector
            and is_supported_release(item)
        ]
        if not matches:
            raise ProviderRejected(
                f"SPHEREx Level-2 detector {detector} requires a local QR3 ePSF calibration"
            )
        if len(matches) > 1:
            raise ProviderRejected(f"Multiple ePSF calibrations match detector {detector}")
        entry = matches[0]
        missing = [key for key in ("path", "hdu") if entry.get(key) is None]
        if missing:
            raise ProviderRejected(
                "ePSF calibration registry entry is missing: " + ", ".join(missing)
            )
        path = Path(entry["path"]).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        if not path.is_file():
            raise ProviderRejected(f"ePSF calibration file is missing: {path}")
        checksum = entry.get("sha256") or entry.get("checksum")
        if not checksum:
            raise ProviderRejected(
                f"ePSF calibration registry entry for detector {detector} is missing a checksum"
            )
        cls._verify_checksum(path, checksum)
        return path, entry["hdu"], entry, "calibration-registry"

    @classmethod
    def _load_table(cls, path, hdu):
        path = Path(path).resolve()
        stamp = (str(path), path.stat().st_mtime_ns, str(hdu))
        if stamp in cls._table_cache:
            return cls._table_cache[stamp]
        with fits.open(path, memmap=False) as hdul:
            table_hdu = hdul[hdu]
            data = np.array(table_hdu.data, copy=True)
            header = table_hdu.header.copy()
        names = {str(name).upper(): str(name) for name in (data.dtype.names or ())}
        required = {"XCENTER", "YCENTER", "EPSF"}
        if not required.issubset(names):
            raise ProviderRejected(
                f"Malformed ePSF table {path}: missing {sorted(required - set(names))}"
            )
        result = (data, header, names)
        cls._table_cache[stamp] = result
        return result

    @staticmethod
    def _tile_positions(shape):
        ny, nx = map(int, shape)
        margin_x = min(5.0, max(0.0, 0.05 * (nx - 1)))
        margin_y = min(5.0, max(0.0, 0.05 * (ny - 1)))
        return [
            ((nx - 1) / 2.0, (ny - 1) / 2.0),
            (margin_x, margin_y),
            (nx - 1 - margin_x, margin_y),
            (margin_x, ny - 1 - margin_y),
            (nx - 1 - margin_x, ny - 1 - margin_y),
        ]

    @staticmethod
    def _parent_position(product, header, position):
        x, y = position
        if product.product_kind == "cutout":
            try:
                return 1.0 + x - float(header["CRPIX1A"]), 1.0 + y - float(header["CRPIX2A"])
            except KeyError as exc:
                raise ProviderRejected("SPHEREx cutout is missing CRPIX1A/CRPIX2A") from exc
        return x, y

    @staticmethod
    def _phases(header, mode):
        if mode == "center":
            return [(0.0, 0.0)]
        oy = int(header.get("OVSMPY", 1))
        ox = int(header.get("OVSMPX", 1))
        return [
            ((iy + 0.5) / oy - 0.5, (ix + 0.5) / ox - 0.5)
            for iy in range(oy)
            for ix in range(ox)
        ]

    def resolve(self, request, config):
        product = request.product
        path, hdu, registry_entry, origin = self._calibration_path(product, config)
        table, epsf_header, names = self._load_table(path, hdu)
        output_sampling = product.sampling
        if output_sampling is None:
            raise ProviderRejected("SPHEREx Level-2 image has no valid angular sampling")
        oversampling = (
            int(epsf_header.get("OVSMPY", 1)),
            int(epsf_header.get("OVSMPX", 1)),
        )
        detcoord = str(epsf_header.get("DETCOORD", "sky")).strip().lower()
        if detcoord not in {"sky", "image"}:
            raise ProviderRejected(
                f"Unsupported ePSF coordinate frame {detcoord!r}; a sky/image-frame calibration is required"
            )
        phases = self._phases(epsf_header, request.phase_sampling)
        positions = self._tile_positions(request.image_shape)
        cache_payload = (
            str(Path(path).resolve()),
            Path(path).stat().st_mtime_ns,
            str(hdu),
            origin,
            None if registry_entry is None else registry_entry.get("release"),
            str(product.detector),
            product.product_kind,
            tuple(map(int, request.image_shape)),
            request.header.get("CRPIX1A"),
            request.header.get("CRPIX2A"),
            output_sampling.matrix_arcsec,
            output_sampling.frame,
            tuple(phases),
            int(config.level2_stamp_size),
            float(config.psf_edge_mass_tolerance),
            float(config.negative_peak_tolerance),
            float(config.negative_mass_tolerance),
            float(config.spatial_r80_tolerance),
            float(config.spatial_ee_tolerance),
            float(config.spatial_l1_tolerance),
            float(config.kernel_centroid_tolerance),
        )
        cache_key = sha256(repr(cache_payload).encode("utf-8")).hexdigest()
        cached = self._ensemble_cache.get(cache_key)
        if cached is not None:
            representative, ensemble = cached
            return ResolvedPSF(representative, ensemble, self.name)
        samples = []
        position_weight = 1.0 / len(positions)
        phase_weight = 1.0 / len(phases)
        xcenters = np.asarray(table[names["XCENTER"]], dtype=float)
        ycenters = np.asarray(table[names["YCENTER"]], dtype=float)
        for position in positions:
            parent_x, parent_y = self._parent_position(product, request.header, position)
            distance = np.hypot(parent_x - xcenters, parent_y - ycenters)
            row_index = int(np.nanargmin(distance))
            epsf = np.asarray(table[names["EPSF"]][row_index], dtype=float)
            if epsf.ndim != 2:
                raise ProviderRejected(f"ePSF row {row_index} is not a 2-D array")
            if not np.all(np.isfinite(epsf)):
                raise ProviderRejected(f"ePSF row {row_index} contains non-finite samples")
            peak = float(np.max(epsf))
            if peak <= 0:
                raise ProviderRejected(f"ePSF row {row_index} has no positive peak")
            negative_mass = float(-np.sum(np.minimum(epsf, 0.0), dtype=np.float64))
            positive_mass = float(np.sum(np.maximum(epsf, 0.0), dtype=np.float64))
            negative_fraction = negative_mass / positive_mass if positive_mass > 0 else np.inf
            if float(np.min(epsf)) < -config.negative_peak_tolerance * peak:
                raise ProviderRejected(f"ePSF row {row_index} contains significant negative samples")
            if negative_fraction > config.negative_mass_tolerance:
                raise ProviderRejected(
                    f"ePSF row {row_index} negative mass fraction {negative_fraction:.3g} exceeds "
                    f"{config.negative_mass_tolerance:.3g}"
                )
            epsf = np.maximum(epsf, 0.0)
            for phase in phases:
                rendered = render_oversampled_psf(
                    epsf,
                    oversampling,
                    phase,
                    output_sampling,
                    stamp_size=config.level2_stamp_size,
                    integration_subsamples=min(8, max(oversampling)),
                )
                model = self.validator(
                    rendered,
                    output_sampling,
                    f"spherex-epsf:{path.name}:row{row_index}",
                    edge_tolerance=config.psf_edge_mass_tolerance,
                    negative_peak_tolerance=config.negative_peak_tolerance,
                    negative_mass_tolerance=config.negative_mass_tolerance,
                    centroid_limit=1.0,
                    frame="image",
                    provenance={
                        "provider": self.name,
                        "origin": origin,
                        "path": str(path),
                        "row": row_index,
                        "binx": int(table[names["BINX"]][row_index]) if "BINX" in names else None,
                        "biny": int(table[names["BINY"]][row_index]) if "BINY" in names else None,
                        "detcoord": detcoord,
                        "negative_mass_removed": negative_mass,
                        "negative_mass_fraction": negative_fraction,
                        "release": None if registry_entry is None else registry_entry.get("release"),
                    },
                )
                samples.append(PSFSample(
                    model,
                    position=tuple(map(float, position)),
                    phase=tuple(map(float, phase)),
                    weight=position_weight * phase_weight,
                    metadata={"row": row_index, "parent_position": (parent_x, parent_y)},
                ))
        ensemble = PSFEnsemble(samples, provenance={
            "provider": self.name,
            "calibration": str(path),
            "oversampling": oversampling,
            "phase_count": len(phases),
            "position_count": len(positions),
        })
        ensemble.spatial_metrics = summarize_spatial_ensemble(
            ensemble,
            r80_tolerance=config.spatial_r80_tolerance,
            ee_tolerance=config.spatial_ee_tolerance,
            l1_tolerance=config.spatial_l1_tolerance,
            centroid_tolerance=config.kernel_centroid_tolerance,
        )
        representative = representative_from_ensemble(
            ensemble,
            f"spherex-epsf-ensemble:D{product.detector}",
            validator=self.validator,
            edge_tolerance=config.psf_edge_mass_tolerance,
        )
        if len(self._ensemble_cache) >= 256:
            self._ensemble_cache.pop(next(iter(self._ensemble_cache)))
        self._ensemble_cache[cache_key] = (representative, ensemble)
        return ResolvedPSF(representative, ensemble, self.name)


@register_provider_factory
def _spherex_provider_factory(*, validator):
    return SpherexEPSFProvider(validator=validator)


__all__ = ["SpherexEPSFProvider", "SpherexProductAdapter"]
