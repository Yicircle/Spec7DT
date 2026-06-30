from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Use the modern NumPy trapezoid integrator with a fallback for older NumPy."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))


def prepare_filter_curve(
    wavelength,
    response,
    *,
    require_positive_response: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return validated wavelength and response arrays sorted by wavelength."""
    wavelength = np.asarray(wavelength, dtype=float)
    response = np.asarray(response, dtype=float)

    if wavelength.ndim != 1 or response.ndim != 1:
        raise ValueError("Wavelength and response must be one-dimensional arrays")
    if wavelength.size != response.size:
        raise ValueError("Wavelength and response arrays must have the same length")
    if wavelength.size < 2:
        raise ValueError("A filter curve must contain at least two samples")
    if not np.all(np.isfinite(wavelength)):
        raise ValueError("Wavelength array contains non-finite values")
    if not np.all(np.isfinite(response)):
        raise ValueError("Response array contains non-finite values")
    if np.any(wavelength <= 0):
        raise ValueError("Wavelength values must be positive")
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    response = response[order]

    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("Wavelength values must be unique")
    if require_positive_response and np.max(response) <= 0:
        raise ValueError("Response curve must contain at least one positive value")

    return wavelength, response


@dataclass(frozen=True)
class FilterProperties:
    """Summary properties derived from a filter response curve."""

    pivot_wavelength: float
    mean_wavelength: float
    peak_wavelength: float
    center_wavelength: float
    fwhm: float

    def as_dict(self) -> Dict[str, float]:
        """Return properties as a plain dictionary."""
        return asdict(self)


class FilterPropertyCalculator:
    """Numerical property calculations for filter response curves."""

    @classmethod
    def calculate(cls, wavelength, response) -> FilterProperties:
        """Calculate all supported response-curve properties."""
        wavelength, response = prepare_filter_curve(
            wavelength,
            response,
            require_positive_response=True,
        )
        left, right = cls._half_max_bounds_prepared(wavelength, response)

        return FilterProperties(
            pivot_wavelength=cls._pivot_wavelength_prepared(wavelength, response),
            mean_wavelength=cls._mean_wavelength_prepared(wavelength, response),
            peak_wavelength=cls._peak_wavelength_prepared(wavelength, response),
            center_wavelength=float((left + right) / 2.0),
            fwhm=float(right - left),
        )

    @staticmethod
    def pivot_wavelength(wavelength, response) -> float:
        """Calculate the pivot wavelength."""
        wavelength, response = prepare_filter_curve(
            wavelength,
            response,
            require_positive_response=True,
        )

        return FilterPropertyCalculator._pivot_wavelength_prepared(wavelength, response)

    @staticmethod
    def mean_wavelength(wavelength, response) -> float:
        """Calculate the response-weighted mean wavelength."""
        wavelength, response = prepare_filter_curve(
            wavelength,
            response,
            require_positive_response=True,
        )

        return FilterPropertyCalculator._mean_wavelength_prepared(wavelength, response)

    @staticmethod
    def peak_wavelength(wavelength, response) -> float:
        """Return the midpoint of all samples tied for maximum response."""
        wavelength, response = prepare_filter_curve(
            wavelength,
            response,
            require_positive_response=True,
        )

        return FilterPropertyCalculator._peak_wavelength_prepared(wavelength, response)

    @classmethod
    def center_wavelength(cls, wavelength, response) -> float:
        """Calculate the midpoint between the half-maximum crossings."""
        left, right = cls._half_max_bounds(wavelength, response)
        return float((left + right) / 2.0)

    @classmethod
    def fwhm(cls, wavelength, response) -> float:
        """Calculate the full width at half maximum."""
        left, right = cls._half_max_bounds(wavelength, response)
        return float(right - left)

    @classmethod
    def _half_max_bounds(cls, wavelength, response) -> Tuple[float, float]:
        """Return the left and right half-maximum crossing wavelengths."""
        wavelength, response = prepare_filter_curve(
            wavelength,
            response,
            require_positive_response=True,
        )
        return cls._half_max_bounds_prepared(wavelength, response)

    @staticmethod
    def _pivot_wavelength_prepared(wavelength: np.ndarray, response: np.ndarray) -> float:
        """Calculate pivot wavelength for a validated curve."""
        numerator = _integrate(response * wavelength, wavelength)
        denominator = _integrate(response / wavelength, wavelength)
        if denominator <= 0:
            raise ValueError("Cannot calculate pivot wavelength from zero response")
        return float(np.sqrt(numerator / denominator))

    @staticmethod
    def _mean_wavelength_prepared(wavelength: np.ndarray, response: np.ndarray) -> float:
        """Calculate mean wavelength for a validated curve."""
        numerator = _integrate(response * wavelength, wavelength)
        denominator = _integrate(response, wavelength)
        if denominator <= 0:
            raise ValueError("Cannot calculate mean wavelength from zero response")
        return float(numerator / denominator)

    @staticmethod
    def _peak_wavelength_prepared(wavelength: np.ndarray, response: np.ndarray) -> float:
        """Calculate peak wavelength for a validated curve."""
        max_response = np.max(response)
        peak_mask = np.isclose(response, max_response, rtol=0.0, atol=0.0)
        return float(np.mean(wavelength[peak_mask]))

    @classmethod
    def _half_max_bounds_prepared(
        cls,
        wavelength: np.ndarray,
        response: np.ndarray,
    ) -> Tuple[float, float]:
        """Return half-maximum bounds for a validated curve."""
        half_max = np.max(response) / 2.0
        above = response >= half_max
        if not np.any(above):
            raise ValueError("No half-maximum samples found")

        left_index = int(np.argmax(above))
        right_index = int(len(above) - 1 - np.argmax(above[::-1]))

        if left_index == 0 or right_index == len(response) - 1:
            raise ValueError("Response curve does not cross half maximum on both sides")

        left = cls._linear_crossing(
            wavelength[left_index - 1],
            response[left_index - 1],
            wavelength[left_index],
            response[left_index],
            half_max,
        )
        right = cls._linear_crossing(
            wavelength[right_index],
            response[right_index],
            wavelength[right_index + 1],
            response[right_index + 1],
            half_max,
        )
        return left, right

    @staticmethod
    def _linear_crossing(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        target: float,
    ) -> float:
        """Interpolate the x-position where a line segment reaches target y."""
        if y1 == y0:
            return float((x0 + x1) / 2.0)
        fraction = (target - y0) / (y1 - y0)
        return float(x0 + (x1 - x0) * fraction)


# Backward-friendly alias for code that imports the old helper name.
CurveProp = FilterPropertyCalculator
