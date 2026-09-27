"""Validate the input rain rate field and bring it to mm/h.

The model expects rain rate in mm/h. Radar and NWP products declare their
units in the CF ``units`` attribute, sometimes in kg m-2 s-1 (that is mm/s):
feeding those values to the model unconverted would silently produce an
empty forecast. This module reads the attribute, converts what it knows,
rejects what it does not, and checks that the values are plausible.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

# Factors to mm/h, keyed by a normalized spelling of the units string.
_FACTORS = {
    "mm/h": 1.0,
    "mm/hr": 1.0,
    "mm/hour": 1.0,
    "mmh-1": 1.0,
    "kgm-2h-1": 1.0,
    "kg/m2/h": 1.0,
    "mm/s": 3600.0,
    "mms-1": 3600.0,
    "kgm-2s-1": 3600.0,
    "kg/m2/s": 3600.0,
    "m/s": 3.6e6,
    "ms-1": 3.6e6,
    "m/h": 1000.0,
    "mh-1": 1000.0,
    "mm/min": 60.0,
    "mmmin-1": 60.0,
}

TRAINED_PAST_STEPS = 6  # frames the released model (IRENE) was trained on: 6 x 5 min

MAX_PLAUSIBLE_MM_H = 1000.0  # well above any radar estimate (60 dBZ is ~205 mm/h)
LOW_MAX_MM_H = 0.5  # a field whose maximum is below this is either dry or in the wrong units


class InputError(ValueError):
    """The input cannot be used as it is."""


class InputUnitsError(InputError):
    """The input units are unknown, or the converted values are not plausible."""


@dataclass
class RainRateInput:
    """Rain rate field in mm/h, with what was done to get there."""

    values: np.ndarray
    units: str
    factor: float
    messages: list[str] = field(default_factory=list)


def _normalize(units: str) -> str:
    u = units.strip().lower()
    u = u.replace("**", "").replace("^", "")
    u = re.sub(r"\s+", "", u)
    u = u.replace("hour-1", "h-1").replace("hr-1", "h-1").replace("sec-1", "s-1")
    u = u.replace("millimeter", "mm").replace("millimetre", "mm")
    return u


def factor_to_mm_h(units: str) -> float:
    """
    Multiplicative factor that brings values in ``units`` to mm/h.

    Raises
    ------
    InputUnitsError
        If the units are not recognized.
    """
    key = _normalize(units)
    if key not in _FACTORS:
        known = ", ".join(sorted({"mm/h", "mm h-1", "kg m-2 h-1", "kg m-2 s-1", "mm/s", "m/s", "mm/min"}))
        raise InputUnitsError(
            f"Unknown rain rate units '{units}'. Known units: {known}. Pass units explicitly to override."
        )
    return _FACTORS[key]


def to_rain_rate_mm_h(data: xr.DataArray | np.ndarray, units: str | None = None) -> RainRateInput:
    """
    Bring a rain rate field to mm/h and check that it is plausible.

    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        The input field. For a DataArray the ``units`` attribute is used
        unless ``units`` is given.
    units : str or None, optional
        Units to assume, overriding the attribute. When neither is
        available, mm/h is assumed and a message says so.

    Returns
    -------
    result : RainRateInput
        Values as float32 in mm/h, with NaN kept for missing data, plus the
        units used, the factor applied and any messages worth showing.

    Raises
    ------
    InputUnitsError
        If the units are unknown, or the converted maximum is not plausible
        (which usually means the ``units`` attribute is wrong).
    """
    messages: list[str] = []
    declared = data.attrs.get("units") if isinstance(data, xr.DataArray) else None
    if units is None:
        units = declared
    elif declared and _normalize(declared) != _normalize(units):
        messages.append(f"Units overridden: file says '{declared}', using '{units}'.")
    if units is None:
        units = "mm/h"
        messages.append("No units attribute on the input: assuming mm/h.")

    factor = factor_to_mm_h(units)
    values = np.asarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.float32)

    negative = values < 0
    if negative.any():
        messages.append(f"{int(negative.sum())} negative values treated as missing.")
        values = np.where(negative, np.nan, values)

    if factor != 1.0:
        values = values * np.float32(factor)
        messages.append(f"Converted from '{units}' to mm/h (factor {factor:g}).")

    finite = values[np.isfinite(values)]
    if finite.size:
        vmax = float(finite.max())
        if vmax > MAX_PLAUSIBLE_MM_H:
            raise InputUnitsError(
                f"Maximum rain rate after conversion is {vmax:.0f} mm/h, above the plausible {MAX_PLAUSIBLE_MM_H:.0f} mm/h: "
                f"the units '{units}' are probably wrong for this file. Pass the correct units explicitly."
            )
        if vmax < LOW_MAX_MM_H:
            messages.append(f"Maximum rain rate is only {vmax:.3f} mm/h: dry input, or units other than '{units}'.")
    else:
        messages.append("Input has no finite values.")

    return RainRateInput(values=values, units=units, factor=factor, messages=messages)


def select_past(data: xr.DataArray, past_steps: int = TRAINED_PAST_STEPS) -> tuple[xr.DataArray, str | None]:
    """
    Keep the last ``past_steps`` frames of the input.

    The model accepts sequences of any length, but its hidden state was
    trained on a fixed number of frames: longer inputs degrade the forecast.

    Returns
    -------
    selected : xr.DataArray
        The last ``past_steps`` frames along the first dimension.
    message : str or None
        What was dropped, or ``None`` when the input had exactly ``past_steps`` frames.

    Raises
    ------
    InputError
        If the input has fewer frames than ``past_steps``.
    """
    if past_steps < 1:
        raise InputError(f"past_steps must be at least 1, got {past_steps}.")
    tdim = data.dims[0]
    total = data.sizes[tdim]
    if total < past_steps:
        raise InputError(f"Need at least {past_steps} timesteps, got {total}.")
    if total == past_steps:
        return data, None
    return data.isel({tdim: slice(total - past_steps, None)}), f"Using the last {past_steps} of {total} timesteps."
