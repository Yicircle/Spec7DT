from __future__ import annotations

import numpy as np
from astropy.wcs import WCS

from .models import ImageProductMetadata, PixelSampling


def sampling_from_header(header):
    try:
        matrix = np.asarray(WCS(header, relax=True).celestial.pixel_scale_matrix, dtype=float) * 3600.0
        return PixelSampling(matrix, frame="image")
    except Exception:
        for key in ("PIXSCALE", "PSCALE"):
            value = header.get(key)
            if value is not None:
                return PixelSampling.isotropic(float(value), frame="image")
    return None


def wcs_metadata_from_header(header):
    """Return a JSON-safe description of the output celestial WCS."""

    try:
        wcs_header = WCS(header, relax=True).celestial.to_header(relax=True)
    except Exception:
        return None
    return {
        str(key): value.item() if isinstance(value, np.generic) else value
        for key, value in wcs_header.items()
    }


def embedded_psf_from_hdul(hdul):
    """Describe embedded PSF representations without instrument assumptions."""

    candidates = []
    for index, hdu in enumerate(hdul):
        name = str(hdu.header.get("EXTNAME", "")).strip()
        if name.upper() not in {"EPSF", "PSF"}:
            continue
        data = getattr(hdu, "data", None)
        representation = "binary_table" if getattr(data, "dtype", None) is not None and getattr(data.dtype, "names", None) else "image"
        candidates.append({
            "type": "table" if representation == "binary_table" else "pixel_model",
            "representation": representation,
            "hdu": name or index,
            "ndim": None if data is None else int(np.ndim(data)),
        })
    if not candidates:
        return None
    return {"type": "embedded_candidates", "candidates": candidates}


class ProductAdapterRegistry:
    def __init__(self):
        self._adapters = []

    def register(self, adapter):
        if adapter not in self._adapters:
            self._adapters.append(adapter)
        return adapter

    @property
    def adapters(self):
        return tuple(self._adapters)

    def probe(self, hdul, filepath, image_index, merged_header):
        matches = []
        for adapter in self._adapters:
            metadata = adapter.probe(hdul, filepath, image_index, merged_header)
            if metadata is not None:
                matches.append((int(getattr(adapter, "priority", 0)), adapter, metadata))
        if not matches:
            return generic_product_metadata(
                filepath,
                image_index,
                merged_header,
                hdul=hdul,
            )
        maximum = max(item[0] for item in matches)
        selected = [item for item in matches if item[0] == maximum]
        if len(selected) > 1:
            names = ", ".join(sorted(type(item[1]).__name__ for item in selected))
            raise ValueError(f"Ambiguous image product adapters: {names}")
        return selected[0][2]


PRODUCT_ADAPTERS = ProductAdapterRegistry()


def register_product_adapter(adapter):
    return PRODUCT_ADAPTERS.register(adapter)


def generic_product_metadata(filepath, image_index, header, *, hdul=None):
    return ImageProductMetadata(
        product_kind="exposure",
        facility=header.get("TELESCOP") or header.get("OBSERVAT"),
        instrument=header.get("INSTRUME"),
        version=header.get("VERSION"),
        source_path=str(filepath),
        hdu=int(image_index),
        sampling=sampling_from_header(header),
        output_wcs=wcs_metadata_from_header(header),
        requires_output_effective_psf=False,
        embedded_psf=None if hdul is None else embedded_psf_from_hdul(hdul),
        provenance={"adapter": "generic-fits"},
    )


def infer_product_metadata(hdul, filepath, image_index, merged_header):
    return PRODUCT_ADAPTERS.probe(hdul, filepath, image_index, merged_header)


def finalize_product_metadata(metadata, *, facility, channel, source_path):
    metadata = ImageProductMetadata.from_value(metadata)
    return metadata.with_identity(
        facility=facility,
        channel=channel,
        source_path=source_path,
    )


__all__ = [
    "PRODUCT_ADAPTERS",
    "finalize_product_metadata",
    "generic_product_metadata",
    "infer_product_metadata",
    "register_product_adapter",
    "sampling_from_header",
]
