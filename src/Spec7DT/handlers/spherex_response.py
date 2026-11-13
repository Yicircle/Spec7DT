from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np


SPHEREX_CALIBRATION_VERSION = "202607"

# Band labels are defined by the nominal wavelengths stored in the 202607
# SPHEREx mosaic SPECTRAL_CHANNELS table.  Response files are always matched
# by (detector, subchannel), never by nearest wavelength.
_BAND_NAMES_BY_DETECTOR = (
    (
        "m753", "m771", "m790", "m809", "m828", "m848", "m868", "m889",
        "m910", "m932", "m955", "m978", "m1001", "m1025", "m1050",
        "m1075", "m1102",
    ),
    (
        "m1113", "m1139", "m1167", "m1195", "m1224", "m1254", "m1284",
        "m1315", "m1346", "m1378", "m1412", "m1446", "m1481", "m1517",
        "m1554", "m1592", "m1631",
    ),
    (
        "m1656", "m1695", "m1735", "m1775", "m1818", "m1861", "m1905",
        "m1949", "m1995", "m2041", "m2089", "m2139", "m2189", "m2240",
        "m2292", "m2345", "m2399",
    ),
    (
        "m2445", "m2506", "m2575", "m2646", "m2719", "m2794", "m2871",
        "m2951", "m3033", "m3118", "m3204", "m3293", "m3383", "m3476",
        "m3570", "m3667", "m3768",
    ),
    (
        "m3833", "m3867", "m3901", "m3935", "m3970", "m4005", "m4040",
        "m4076", "m4112", "m4148", "m4184", "m4220", "m4257", "m4294",
        "m4331", "m4368", "m4405",
    ),
    (
        "m4435", "m4467", "m4500", "m4533", "m4566", "m4600", "m4635",
        "m4669", "m4704", "m4739", "m4773", "m4808", "m4843", "m4879",
        "m4915", "m4951", "m4988",
    ),
)

_CHANNEL_BY_BAND = {
    band.lower(): (detector, subchannel)
    for detector, bands in enumerate(_BAND_NAMES_BY_DETECTOR, start=1)
    for subchannel, band in enumerate(bands, start=1)
}


def band_name_for_channel(detector: int, subchannel: int) -> str:
    """Return the 202607 mosaic band label for a detector/subchannel pair."""
    try:
        return _BAND_NAMES_BY_DETECTOR[int(detector) - 1][int(subchannel) - 1]
    except (IndexError, TypeError, ValueError) as exc:
        raise KeyError(
            f"Invalid SPHEREx detector/subchannel: {detector}/{subchannel}"
        ) from exc


def channel_for_band(name: str) -> tuple[int, int]:
    """Resolve a canonical or short SPHEREx band name to detector/subchannel."""
    band = str(name).strip().split("/")[-1].split(".")[-1]
    try:
        return _CHANNEL_BY_BAND[band.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown SPHEREx 202607 band '{name}'") from exc


def _default_response_dir():
    return (
        resources.files("Spec7DT")
        .joinpath("reference")
        .joinpath("spherex")
        .joinpath(SPHEREX_CALIBRATION_VERSION)
        .joinpath("MeanAllPixel")
        .joinpath("SubChannel17")
    )


def _resolve_response_dir(response_root=None):
    if response_root is None:
        response_dir = _default_response_dir()
    else:
        response_dir = Path(response_root).expanduser()
        nested = response_dir / "MeanAllPixel" / "SubChannel17"
        if nested.is_dir():
            response_dir = nested

    if not response_dir.is_dir():
        raise FileNotFoundError(
            "SPHEREx MeanAllPixel/SubChannel17 response directory not found: "
            f"{response_dir}"
        )
    return response_dir


def load_spherex_response(
    detector: int,
    subchannel: int,
    *,
    response_root=None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one official 17-subchannel all-pixel response curve.

    Wavelengths are returned in Angstrom.  Negative laboratory noise samples
    are clipped to zero because downstream code treats the values as physical
    transmission weights.
    """
    detector = int(detector)
    subchannel = int(subchannel)
    band = band_name_for_channel(detector, subchannel)
    response_dir = _resolve_response_dir(response_root)
    filename = (
        f"FiducialSpec_Band{detector}_SubCh{subchannel:02d}_17_Allpix.csv"
    )
    resource = response_dir.joinpath(f"Band{detector}").joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(
            f"SPHEREx response file missing for detector {detector}, "
            f"subchannel {subchannel}: {resource}"
        )

    with resources.as_file(resource) as filepath:
        table = np.genfromtxt(filepath, delimiter=",", names=True)

    names = set(table.dtype.names or ())
    required = {"Wavelength_um", "NormTransmission"}
    if not required.issubset(names):
        raise ValueError(
            f"Malformed SPHEREx response file {resource}: expected {sorted(required)}"
        )

    wavelength = np.asarray(table["Wavelength_um"], dtype=float) * 1.0e4
    response = np.clip(np.asarray(table["NormTransmission"], dtype=float), 0.0, None)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError(f"SPHEREx response {resource} has insufficient samples")
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(response)):
        raise ValueError(f"SPHEREx response {resource} contains non-finite values")
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    response = response[order]
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError(f"SPHEREx response {resource} wavelengths are not increasing")
    if np.max(response) <= 0:
        raise ValueError(f"SPHEREx response {resource} has no positive transmission")
    return wavelength, response, band


def register_spherex_filter(
    registry_cls,
    *,
    detector: int,
    subchannel: int,
    band_name: str | None = None,
    response_root=None,
):
    """Register and return one SPHEREx response in a Filters-like registry."""
    expected_band = band_name_for_channel(detector, subchannel)
    band_name = expected_band if band_name is None else str(band_name)
    if band_name.lower() != expected_band.lower():
        raise ValueError(
            "SPHEREx cube channel does not match the 202607 calibration index: "
            f"detector={detector}, subchannel={subchannel}, "
            f"cube={band_name}, expected={expected_band}"
        )

    canonical_name = f"SPHEREx.{band_name}"
    try:
        return registry_cls.get_filter(name=canonical_name)
    except (KeyError, ValueError):
        pass

    wavelength, response, _ = load_spherex_response(
        detector,
        subchannel,
        response_root=response_root,
    )
    registry_cls.add_custom(
        canonical_name,
        wavelength,
        response,
        unit_type="photon",
        description=(
            f"IRSA SPHEREx {SPHEREX_CALIBRATION_VERSION} MeanAllPixel "
            f"SubChannel17 detector {detector} subchannel {subchannel}"
        ),
        announce=False,
    )
    return registry_cls.get_filter(name=canonical_name)


def ensure_local_spherex_filter(registry_cls, name: str, facility: str | None):
    """Return a packaged SPHEREx filter, or ``None`` for unrelated requests."""
    if facility is None or str(facility).strip().lower() != "spherex":
        return None
    detector, subchannel = channel_for_band(name)
    return register_spherex_filter(
        registry_cls,
        detector=detector,
        subchannel=subchannel,
    )


__all__ = [
    "SPHEREX_CALIBRATION_VERSION",
    "band_name_for_channel",
    "channel_for_band",
    "ensure_local_spherex_filter",
    "load_spherex_response",
    "register_spherex_filter",
]
