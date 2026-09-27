"""Keep forecast requests within the horizon the model was trained on.

The decoder can be unrolled for any number of steps, but beyond the
training horizon it produces states it has never been trained on: the
numbers keep coming and nothing says they are guesses. Requests are capped
at the trained horizon unless the cap is raised explicitly; forecasts past
the horizon are flagged in the output.
"""

DEFAULT_TRAINED_FORECAST_STEPS = 12  # the released model (IRENE): 12 x 5 min


class HorizonError(ValueError):
    """The requested number of forecast steps is above the allowed cap."""


def trained_forecast_steps(model) -> int:
    """Forecast steps the model was trained on, from its hyperparameters."""
    value = getattr(getattr(model, "hparams", None), "forecast_steps", None)
    return int(value) if isinstance(value, int) and value > 0 else DEFAULT_TRAINED_FORECAST_STEPS


def check_forecast_steps(forecast_steps: int, trained: int, max_forecast_steps: int | None = None) -> str | None:
    """
    Validate a request against the trained horizon and the configured cap.

    Parameters
    ----------
    forecast_steps : int
        Steps requested.
    trained : int
        Steps the model was trained on.
    max_forecast_steps : int or None, optional
        Cap on the request. ``None`` means the trained horizon.

    Returns
    -------
    warning : str or None
        A message when the request goes past the trained horizon but is
        within the cap, otherwise ``None``.

    Raises
    ------
    HorizonError
        If ``forecast_steps`` is below 1 or above the cap.
    """
    cap = trained if max_forecast_steps is None else max_forecast_steps
    if forecast_steps < 1:
        raise HorizonError(f"forecast_steps must be at least 1, got {forecast_steps}.")
    if forecast_steps > cap:
        hint = (
            f"Raise max_forecast_steps to go past the {trained} steps the model was trained on."
            if cap == trained
            else f"The configured cap is {cap}."
        )
        raise HorizonError(f"forecast_steps={forecast_steps} is above the allowed {cap}. {hint}")
    if forecast_steps > trained:
        return (
            f"Forecast steps {trained + 1}-{forecast_steps} are beyond the {trained} steps the model was trained on: "
            "treat them as extrapolation."
        )
    return None
