from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np


def _matrix_tuple(value) -> tuple[tuple[float, float], tuple[float, float]]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("Pixel sampling matrix must be a finite 2x2 array")
    if abs(float(np.linalg.det(matrix))) <= np.finfo(float).eps:
        raise ValueError("Pixel sampling matrix must be invertible")
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True)
class PixelSampling:
    """Angular pixel transform in arcsec per pixel."""

    matrix_arcsec: tuple[tuple[float, float], tuple[float, float]]
    frame: Literal["image", "sky", "detector"] = "image"

    def __post_init__(self):
        object.__setattr__(self, "matrix_arcsec", _matrix_tuple(self.matrix_arcsec))
        if self.frame not in {"image", "sky", "detector"}:
            raise ValueError("PixelSampling frame must be image, sky, or detector")

    @classmethod
    def isotropic(cls, scale_arcsec: float, frame: str = "image"):
        scale = float(scale_arcsec)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Pixel scale must be finite and positive")
        return cls(((scale, 0.0), (0.0, scale)), frame=frame)

    @property
    def equivalent_scale_arcsec(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(np.asarray(self.matrix_arcsec)))))

    def to_dict(self):
        return {"matrix_arcsec": [list(row) for row in self.matrix_arcsec], "frame": self.frame}

    @classmethod
    def from_value(cls, value, *, default_scale=None, default_frame="image"):
        if isinstance(value, cls):
            return value
        if value is None:
            if default_scale is None:
                raise ValueError("Pixel sampling or a default scale is required")
            return cls.isotropic(default_scale, frame=default_frame)
        if isinstance(value, Mapping):
            matrix = value.get("matrix_arcsec") or value.get("matrix")
            if matrix is None:
                scale = value.get("pixel_scale_arcsec", value.get("scale_arcsec"))
                return cls.isotropic(scale, frame=str(value.get("frame", default_frame)))
            return cls(matrix, frame=str(value.get("frame", default_frame)))
        if np.isscalar(value):
            return cls.isotropic(float(value), frame=default_frame)
        return cls(value, frame=default_frame)


def make_product_id(source_path, hdu=None, plane=None, extra=None) -> str:
    """Return a stable local identity without hashing a potentially huge FITS file."""

    source = "<memory>" if source_path is None else str(Path(source_path).expanduser().resolve())
    payload = "|".join((source, str(hdu), str(plane), str(extra)))
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ImageProductMetadata:
    product_kind: Literal["exposure", "cutout", "coadd"] = "exposure"
    facility: str | None = None
    instrument: str | None = None
    release: str | None = None
    version: str | None = None
    detector: str | int | None = None
    channel: str | None = None
    source_path: str | None = None
    hdu: str | int | None = None
    plane: int | None = None
    product_id: str | None = None
    product_group_id: str | None = None
    sampling: PixelSampling | None = None
    output_wcs: Mapping[str, Any] | None = None
    requires_output_effective_psf: bool = False
    embedded_psf: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.product_kind not in {"exposure", "cutout", "coadd"}:
            raise ValueError("product_kind must be exposure, cutout, or coadd")
        sampling = self.sampling
        if sampling is not None and not isinstance(sampling, PixelSampling):
            sampling = PixelSampling.from_value(sampling)
            object.__setattr__(self, "sampling", sampling)
        if self.product_id is None:
            object.__setattr__(
                self,
                "product_id",
                make_product_id(self.source_path, self.hdu, self.plane, self.channel),
            )
        object.__setattr__(self, "provenance", dict(self.provenance or {}))
        if self.output_wcs is not None:
            object.__setattr__(self, "output_wcs", dict(self.output_wcs))
        if self.embedded_psf is not None:
            object.__setattr__(self, "embedded_psf", dict(self.embedded_psf))

    def with_identity(self, *, facility=None, channel=None, source_path=None):
        identity_was_incomplete = self.source_path is None or self.channel is None
        values = {
            "facility": self.facility or (None if facility is None else str(facility)),
            "channel": self.channel or (None if channel is None else str(channel)),
            "source_path": self.source_path or (None if source_path is None else str(source_path)),
        }
        if self.product_id is None or identity_was_incomplete:
            values["product_id"] = make_product_id(
                values["source_path"], self.hdu, self.plane, values["channel"]
            )
        return replace(self, **values)

    def to_dict(self):
        return {
            "product_kind": self.product_kind,
            "facility": self.facility,
            "instrument": self.instrument,
            "release": self.release,
            "version": self.version,
            "detector": self.detector,
            "channel": self.channel,
            "source_path": self.source_path,
            "hdu": self.hdu,
            "plane": self.plane,
            "product_id": self.product_id,
            "product_group_id": self.product_group_id,
            "sampling": None if self.sampling is None else self.sampling.to_dict(),
            "output_wcs": None if self.output_wcs is None else dict(self.output_wcs),
            "requires_output_effective_psf": bool(self.requires_output_effective_psf),
            "embedded_psf": None if self.embedded_psf is None else dict(self.embedded_psf),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_value(cls, value=None, **defaults):
        if isinstance(value, cls):
            return value
        data = {} if value is None else dict(value)
        for key, item in defaults.items():
            if item is not None:
                data.setdefault(key, item)
        if data.get("sampling") is not None:
            data["sampling"] = PixelSampling.from_value(data["sampling"])
        return cls(**data)


@dataclass
class EffectivePSF:
    """Normalized, pixel-integrated PSF and its physical sampling."""

    data: np.ndarray
    pixel_scale_arcsec: float
    fwhm_arcsec: float
    r50_arcsec: float
    r80_arcsec: float
    source: str
    frame: str = "image"
    quality: dict[str, Any] = field(default_factory=dict)
    sampling: PixelSampling | None = None
    validity_region: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.float32)
        self.pixel_scale_arcsec = float(self.pixel_scale_arcsec)
        self.sampling = PixelSampling.from_value(
            self.sampling,
            default_scale=self.pixel_scale_arcsec,
            default_frame=self.frame if self.frame in {"image", "sky", "detector"} else "image",
        )
        self.quality = dict(self.quality or {})
        self.provenance = dict(self.provenance or {})


@dataclass(frozen=True)
class PSFRequest:
    product: ImageProductMetadata
    galaxy: str
    observatory: str
    band: str
    image_shape: tuple[int, int]
    header: Any
    image_data: np.ndarray | None = None
    tile_bounds: tuple[float, float, float, float] | None = None
    phase_sampling: Literal["native", "center"] = "native"


@dataclass
class PSFSample:
    model: EffectivePSF
    position: tuple[float, float] | None = None
    phase: tuple[float, float] | None = None
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PSFEnsemble:
    samples: list[PSFSample]
    spatial_metrics: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.samples:
            raise ValueError("A PSF ensemble must contain at least one sample")
        weights = np.asarray([sample.weight for sample in self.samples], dtype=float)
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("PSF ensemble weights must be finite and positive")
        weights /= np.sum(weights)
        for sample, weight in zip(self.samples, weights):
            sample.weight = float(weight)


@dataclass
class ResolvedPSF:
    representative: EffectivePSF
    ensemble: PSFEnsemble
    provider: str
    diagnostics: list[str] = field(default_factory=list)


@runtime_checkable
class PSFField(Protocol):
    """A PSF model evaluable at an image position and sub-pixel phase."""

    def evaluate(
        self,
        position: tuple[float, float] | None = None,
        phase: tuple[float, float] | None = None,
    ) -> EffectivePSF: ...


@runtime_checkable
class PSFProvider(Protocol):
    name: str

    def match(self, request: PSFRequest, config) -> int: ...

    def resolve(self, request: PSFRequest, config) -> ResolvedPSF: ...


@dataclass(frozen=True)
class StationaryPSFField:
    model: EffectivePSF

    def evaluate(self, position=None, phase=None):
        return self.model

