import numpy as np
import pytest
import xarray as xr

from convgru_ensemble.inputs import InputUnitsError, factor_to_mm_h, to_rain_rate_mm_h


def _field(values, units=None):
    da = xr.DataArray(np.asarray(values, dtype=np.float32)[None, :, None], dims=["time", "y", "x"])
    if units is not None:
        da.attrs["units"] = units
    return da


@pytest.mark.parametrize(
    "units, factor",
    [
        ("mm/h", 1.0),
        ("mm h-1", 1.0),
        ("mm h^-1", 1.0),
        ("mm hr-1", 1.0),
        ("kg m-2 h-1", 1.0),
        ("kg m**-2 h**-1", 1.0),
        ("mm/s", 3600.0),
        ("kg m-2 s-1", 3600.0),
        ("m s-1", 3.6e6),
        ("mm/min", 60.0),
    ],
)
def test_known_units(units, factor):
    assert factor_to_mm_h(units) == factor


def test_unknown_units_are_rejected():
    with pytest.raises(InputUnitsError, match="Unknown rain rate units 'dBZ'"):
        factor_to_mm_h("dBZ")


def test_mm_h_passes_unchanged():
    out = to_rain_rate_mm_h(_field([0.0, 2.5, 80.0], "mm/h"))
    np.testing.assert_array_equal(out.values.ravel(), [0.0, 2.5, 80.0])
    assert out.factor == 1.0 and out.messages == []
    assert out.values.dtype == np.float32


def test_per_second_is_converted():
    out = to_rain_rate_mm_h(_field([0.0, 0.001, 0.02], "kg m-2 s-1"))
    np.testing.assert_allclose(out.values.ravel(), [0.0, 3.6, 72.0], rtol=1e-5)
    assert out.factor == 3600.0
    assert any("Converted" in m for m in out.messages)


def test_missing_units_assume_mm_h_with_message():
    out = to_rain_rate_mm_h(_field([1.0, 5.0]))
    assert out.units == "mm/h"
    assert any("assuming mm/h" in m for m in out.messages)


def test_override_wins_over_attribute():
    out = to_rain_rate_mm_h(_field([1.0, 100.0], "kg m-2 s-1"), units="mm/h")
    np.testing.assert_array_equal(out.values.ravel(), [1.0, 100.0])
    assert any("overridden" in m for m in out.messages)


def test_wrong_units_attribute_is_caught_by_plausibility():
    # Values are really mm/h but the file claims per second: conversion gives an impossible field.
    with pytest.raises(InputUnitsError, match="probably wrong"):
        to_rain_rate_mm_h(_field([0.0, 113.0], "kg m-2 s-1"))


def test_negative_values_become_missing():
    out = to_rain_rate_mm_h(_field([-9999.0, 0.0, 3.0], "mm/h"))
    assert np.isnan(out.values.ravel()[0])
    assert any("negative" in m for m in out.messages)


def test_nan_is_preserved_and_dry_input_is_flagged():
    out = to_rain_rate_mm_h(_field([np.nan, 0.0, 0.1], "mm/h"))
    assert np.isnan(out.values.ravel()[0])
    assert any("dry input" in m for m in out.messages)


def test_plain_array_input():
    out = to_rain_rate_mm_h(np.array([[[0.01]]]), units="mm/s")
    assert out.values.ravel()[0] == pytest.approx(36.0)
