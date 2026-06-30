from __future__ import annotations

import warnings

from Spec7DT.handlers import (
    FilterCurve,
    FilterProperties,
    FilterPropertyCalculator,
    Filters as _CoreFilters,
)


class Observatories:
    """Compatibility observatory name list used by older viewOut parsers."""

    def __init__(self):
        self.optical_obs = self._opticals()
        self.ir_obs = self._infrareds()
        self.uv_obs = self._ultraviolet()
        self.radio_obs = self._radio()
        self.observatories = list(set(self.optical_obs + self.ir_obs + self.uv_obs + self.radio_obs))

    def _opticals(self):
        return ["HST", "SDSS", "PS1", "CFHT", "DECam", "DES", "LSST", "Pan-STARRS", "Subaru", "7DT", "SkyMapper", "sloan"]

    def _infrareds(self):
        return ["WISE", "Spitzer", "Herschel", "JWST", "VISTA", "UKIDSS", "2MASS", "SPHEREx", "SPIRE", "PACS", "spire", "pacs", "wise"]

    def _ultraviolet(self):
        return ["GALEX", "HST", "FUSE", "galex"]

    def _radio(self):
        return ["VLA", "ALMA", "LOFAR", "SKA", "MeerKAT", "GMRT"]

    @classmethod
    def get_observatories(cls):
        return cls().observatories


class CurveProp:
    """Compatibility wrapper around Spec7DT filter property calculations."""

    pivot_wavelength = staticmethod(FilterPropertyCalculator.pivot_wavelength)
    mean_wavelength = staticmethod(FilterPropertyCalculator.mean_wavelength)
    peak_wavelength = staticmethod(FilterPropertyCalculator.peak_wavelength)
    center_wavelength = staticmethod(FilterPropertyCalculator.center_wavelength)
    fwhm = staticmethod(FilterPropertyCalculator.fwhm)
    FWHM = staticmethod(FilterPropertyCalculator.fwhm)


class Filters(_CoreFilters):
    """Deprecated viewOut filter registry wrapper.

    The canonical implementation lives in Spec7DT.handlers.filter_handler.
    This class preserves old notebook imports and the legacy positional
    get_filter_curve(observatory, band) calling style.
    """

    def __init__(self):
        warnings.warn(
            "viewOut.filter_property.Filters is deprecated; use Spec7DT.handlers.Filters.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

    @classmethod
    def get_filter_curve(cls, name=None, facility=None, instrument=None):
        if facility is not None and instrument is None:
            try:
                return super().get_filter_curve(name=name, facility=facility)
            except KeyError:
                return super().get_filter_curve(name=facility, facility=name)
        return super().get_filter_curve(name=name, facility=facility, instrument=instrument)

    @classmethod
    def get_filter(cls, name=None, facility=None, instrument=None):
        if facility is not None and instrument is None:
            try:
                return super().get_filter(name=name, facility=facility)
            except KeyError:
                return super().get_filter(name=facility, facility=name)
        return super().get_filter(name=name, facility=facility, instrument=instrument)


__all__ = [
    "CurveProp",
    "FilterCurve",
    "FilterProperties",
    "FilterPropertyCalculator",
    "Filters",
    "Observatories",
]
