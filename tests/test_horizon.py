from types import SimpleNamespace

import pytest

from convgru_ensemble.horizon import HorizonError, check_forecast_steps, trained_forecast_steps


def test_trained_steps_come_from_hparams():
    assert trained_forecast_steps(SimpleNamespace(hparams=SimpleNamespace(forecast_steps=9))) == 9
    assert trained_forecast_steps(SimpleNamespace(hparams=SimpleNamespace(forecast_steps=None))) == 12
    assert trained_forecast_steps(object()) == 12


def test_within_horizon_is_silent():
    assert check_forecast_steps(12, 12) is None
    assert check_forecast_steps(1, 12) is None


def test_beyond_horizon_is_rejected_by_default():
    with pytest.raises(HorizonError, match="above the allowed 12"):
        check_forecast_steps(13, 12)


def test_raised_cap_allows_with_warning():
    warning = check_forecast_steps(24, 12, max_forecast_steps=48)
    assert "13-24" in warning and "extrapolation" in warning
    with pytest.raises(HorizonError, match="configured cap is 48"):
        check_forecast_steps(49, 12, max_forecast_steps=48)


def test_zero_steps_is_rejected():
    with pytest.raises(HorizonError):
        check_forecast_steps(0, 12)
