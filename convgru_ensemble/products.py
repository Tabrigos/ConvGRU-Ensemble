"""Derived products from an ensemble forecast.

Ten members times twelve steps over Italy is 800 MB nobody wants to look
at. What a user looks at are the summaries: the ensemble mean and median,
percentiles, the probability of exceeding a rain rate, and accumulations
over the next 30 or 60 minutes. All of them are computed here from the
dataset written by the CLI or the API, and keep its georeference.
"""

import numpy as np
import xarray as xr

from .output import DEFAULT_TIMESTEP

DEFAULT_THRESHOLDS_MM_H = (0.1, 1.0, 5.0, 20.0)
DEFAULT_PERCENTILES = (10, 50, 90)
DEFAULT_ACCUMULATIONS_MIN = (30, 60)

FORECAST_VAR = "precipitation_forecast"
MEMBER_DIM = "ensemble_member"
STEP_DIM = "forecast_step"


def timestep_minutes(forecast: xr.Dataset) -> float:
    """Minutes between forecast steps, from the attributes, the valid times or the model default."""
    value = forecast.attrs.get("timestep_minutes")
    if value is not None:
        return float(value)
    if "forecast_time" in forecast.coords and forecast.sizes.get(STEP_DIM, 0) >= 2:
        t = forecast["forecast_time"].values.astype("datetime64[ns]")
        return float((t[1] - t[0]) / np.timedelta64(1, "m"))
    return float(DEFAULT_TIMESTEP / np.timedelta64(1, "m"))


def _forecast(forecast: xr.Dataset) -> xr.DataArray:
    if FORECAST_VAR not in forecast:
        raise ValueError(f"Dataset has no '{FORECAST_VAR}' variable; got {list(forecast.data_vars)}")
    da = forecast[FORECAST_VAR]
    if da.dims[:2] != (MEMBER_DIM, STEP_DIM):
        raise ValueError(f"Expected dims ({MEMBER_DIM}, {STEP_DIM}, y, x), got {da.dims}")
    return da


def _carry_georeference(source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Copy the grid mapping variable and the georeference attributes to the products."""
    if "crs" in source:
        target["crs"] = source["crs"]
        for name, var in target.data_vars.items():
            if name != "crs" and len(var.dims) >= 2:
                var.attrs.setdefault("grid_mapping", "crs")
    for key in ("model", "forecast_reference_time", "timestep_minutes", "Conventions", "trained_forecast_steps"):
        if key in source.attrs:
            target.attrs[key] = source.attrs[key]
    if "forecast_reference_time" in source.coords:
        target = target.assign_coords(forecast_reference_time=source["forecast_reference_time"])
    return target


def ensemble_products(
    forecast: xr.Dataset,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS_MM_H,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    accumulations_min: tuple[float, ...] = DEFAULT_ACCUMULATIONS_MIN,
) -> xr.Dataset:
    """
    Summarize an ensemble forecast into the products a user looks at.

    Parameters
    ----------
    forecast : xr.Dataset
        Dataset written by ``convgru-ensemble predict`` or by the API, with
        ``precipitation_forecast (ensemble_member, forecast_step, y, x)`` in
        mm/h and, when available, the georeference.
    thresholds : tuple of float, optional
        Rain rates in mm/h for the probability of exceedance.
    percentiles : tuple of float, optional
        Percentiles of the ensemble to compute, in 0-100.
    accumulations_min : tuple of float, optional
        Windows in minutes, from the first step, for the accumulations.
        Windows longer than the forecast are skipped.

    Returns
    -------
    products : xr.Dataset
        ``rain_rate_mean``, ``rain_rate_median``, ``rain_rate_spread``
        (standard deviation across members), ``rain_rate_percentile``
        (dim ``percentile``), ``probability_exceeding`` (dim ``threshold``,
        in 0-1), ``accumulation_mean`` and ``accumulation_percentile``
        (dim ``accumulation``, in mm), all on the input grid.
    """
    da = _forecast(forecast)
    members = da.sizes[MEMBER_DIM]
    step_h = timestep_minutes(forecast) / 60.0
    rate_attrs = {"units": "mm h-1"}

    # The forecast has no NaN (missing input pixels are "no rain"), so the plain numpy
    # reductions are used: the NaN-aware ones are an order of magnitude slower.
    products = xr.Dataset(
        {
            "rain_rate_mean": da.mean(MEMBER_DIM, skipna=False).assign_attrs(
                rate_attrs, long_name="Ensemble mean rain rate"
            ),
            "rain_rate_median": da.median(MEMBER_DIM, skipna=False).assign_attrs(
                rate_attrs, long_name="Ensemble median rain rate"
            ),
            "rain_rate_spread": da.std(MEMBER_DIM, skipna=False).assign_attrs(
                rate_attrs, long_name="Ensemble spread (standard deviation across members)"
            ),
        }
    )

    if percentiles:
        pct = da.quantile([p / 100.0 for p in percentiles], dim=MEMBER_DIM, skipna=False)
        pct = pct.rename({"quantile": "percentile"}).assign_coords(percentile=list(percentiles)).astype(da.dtype)
        products["rain_rate_percentile"] = pct.assign_attrs(rate_attrs, long_name="Ensemble percentile of rain rate")
        products["percentile"].attrs.update(long_name="Ensemble percentile, 0-100")

    if thresholds:
        prob = xr.concat([(da >= t).sum(MEMBER_DIM) / members for t in thresholds], dim="threshold")
        prob = prob.assign_coords(threshold=list(thresholds)).astype(np.float32)
        products["probability_exceeding"] = prob.assign_attrs(
            units="1", long_name="Fraction of ensemble members at or above the threshold rain rate"
        )
        products["threshold"].attrs.update(units="mm h-1", long_name="Rain rate threshold")

    windows = [w for w in accumulations_min if w <= da.sizes[STEP_DIM] * step_h * 60 + 1e-9]
    if windows:
        acc = []
        for w in windows:
            n = int(round(w / (step_h * 60)))
            acc.append((da.isel({STEP_DIM: slice(0, n)}) * step_h).sum(STEP_DIM))
        acc = xr.concat(acc, dim="accumulation").assign_coords(accumulation=[int(round(w)) for w in windows])
        products["accumulation_mean"] = acc.mean(MEMBER_DIM, skipna=False).assign_attrs(
            units="mm", long_name="Ensemble mean accumulation from the first forecast step"
        )
        if percentiles:
            acc_pct = acc.quantile([p / 100.0 for p in percentiles], dim=MEMBER_DIM, skipna=False)
            products["accumulation_percentile"] = (
                acc_pct.rename({"quantile": "percentile"})
                .assign_coords(percentile=list(percentiles))
                .astype(da.dtype)
                .assign_attrs(units="mm", long_name="Ensemble percentile of the accumulation")
            )
        # No CF time units on this coordinate: xarray would decode it as a timedelta on reading.
        products["accumulation"].attrs.update(long_name="Accumulation window from the first step, in minutes")

    products.attrs.update(
        {
            "ensemble_size": members,
            "forecast_steps": da.sizes[STEP_DIM],
            "timestep_minutes": step_h * 60,
        }
    )
    return _carry_georeference(forecast, products)
