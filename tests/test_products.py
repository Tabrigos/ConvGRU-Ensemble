import numpy as np
import pytest
import xarray as xr

from convgru_ensemble.output import build_forecast_dataset
from convgru_ensemble.products import ensemble_products, timestep_minutes


def _forecast(members=4, steps=12, h=8, w=10, georeferenced=True):
    rng = np.random.default_rng(0)
    preds = rng.random((members, steps, h, w), dtype=np.float32) * 10
    preds[0] = 0.0  # one dry member
    preds[1, :, 0, 0] = 50.0  # one wet pixel in one member
    if georeferenced:
        y = 649_500.0 - 1000.0 * np.arange(h)
        x = -599_500.0 + 1000.0 * np.arange(w)
        coords = {
            "y": ("y", y),
            "x": ("x", x),
            "time": ("time", np.datetime64("2025-08-28T21:55") + np.timedelta64(5, "m") * np.arange(6)),
        }
        source = xr.DataArray(np.zeros((6, h, w), dtype=np.float32), dims=["time", "y", "x"], coords=coords)
    else:
        source = xr.DataArray(np.zeros((6, h, w), dtype=np.float32), dims=["time", "y", "x"])
    return preds, build_forecast_dataset(preds, source)


def test_mean_median_spread_and_percentiles():
    preds, forecast = _forecast()
    products = ensemble_products(forecast)
    np.testing.assert_allclose(products["rain_rate_mean"].values, preds.mean(0), rtol=1e-6)
    np.testing.assert_allclose(products["rain_rate_median"].values, np.median(preds, axis=0), rtol=1e-6)
    np.testing.assert_allclose(products["rain_rate_spread"].values, preds.std(0), rtol=1e-5)
    p90 = products["rain_rate_percentile"].sel(percentile=90).values
    np.testing.assert_allclose(p90, np.percentile(preds, 90, axis=0), rtol=1e-5)
    assert products["rain_rate_mean"].dims == ("forecast_step", "y", "x")


def test_probability_of_exceedance():
    preds, forecast = _forecast()
    products = ensemble_products(forecast, thresholds=(1.0, 20.0))
    prob = products["probability_exceeding"]
    assert list(prob["threshold"].values) == [1.0, 20.0]
    np.testing.assert_allclose(prob.sel(threshold=1.0).values, (preds >= 1.0).mean(0), rtol=1e-6)
    # the wet pixel of one member out of four: probability 0.25 at 20 mm/h, at every step
    np.testing.assert_allclose(prob.sel(threshold=20.0).values[:, 0, 0], 0.25)
    assert float(prob.max()) <= 1.0 and float(prob.min()) >= 0.0


def test_accumulations_use_the_timestep():
    preds, forecast = _forecast(steps=12)
    products = ensemble_products(forecast, accumulations_min=(30, 60, 120))
    # 120 minutes is longer than the 60-minute forecast: skipped
    assert list(products["accumulation"].values) == [30, 60]
    expected_60 = (preds * (5 / 60)).sum(1).mean(0)
    np.testing.assert_allclose(products["accumulation_mean"].sel(accumulation=60).values, expected_60, rtol=1e-5)
    expected_30 = (preds[:, :6] * (5 / 60)).sum(1).mean(0)
    np.testing.assert_allclose(products["accumulation_mean"].sel(accumulation=30).values, expected_30, rtol=1e-5)
    assert products["accumulation_mean"].attrs["units"] == "mm"
    assert products["accumulation_percentile"].sel(percentile=90).shape == (2, 8, 10)


def test_georeference_is_carried_over():
    _, forecast = _forecast()
    products = ensemble_products(forecast)
    assert "crs" in products
    np.testing.assert_array_equal(products["x"].values, forecast["x"].values)
    assert products["rain_rate_mean"].attrs["grid_mapping"] == "crs"
    assert "forecast_time" in products.coords
    assert products.attrs["timestep_minutes"] == 5.0


def test_products_without_georeference_still_work():
    _, forecast = _forecast(georeferenced=False)
    products = ensemble_products(forecast)
    assert "crs" not in products
    assert timestep_minutes(forecast) == 5.0
    assert products["rain_rate_mean"].shape == (12, 8, 10)


def test_products_netcdf_roundtrip(tmp_path):
    _, forecast = _forecast()
    products = ensemble_products(forecast)
    path = tmp_path / "products.nc"
    products.to_netcdf(path)
    back = xr.open_dataset(path)
    # plain integer coordinates: selecting by value must work after reading
    assert back["accumulation_mean"].sel(accumulation=60).shape == (8, 10)
    assert back["rain_rate_percentile"].sel(percentile=90).shape == (12, 8, 10)
    assert back["probability_exceeding"].sel(threshold=5.0).shape == (12, 8, 10)
    assert back["crs"].attrs["grid_mapping_name"] == "transverse_mercator"


def test_rejects_wrong_input():
    with pytest.raises(ValueError):
        ensemble_products(xr.Dataset({"other": xr.DataArray(np.zeros((2, 2)), dims=["y", "x"])}))


def test_geotiff_export(tmp_path):
    pytest.importorskip("rasterio")
    import rasterio

    from convgru_ensemble.geotiff import write_products_geotiffs

    _, forecast = _forecast()
    products = ensemble_products(forecast, thresholds=(1.0,), percentiles=(90,), accumulations_min=(60,))
    paths = write_products_geotiffs(products, tmp_path / "tif")
    names = sorted(p.name for p in paths)
    assert names == sorted(
        [
            "rain_rate_mean.tif",
            "rain_rate_median.tif",
            "rain_rate_spread.tif",
            "rain_rate_percentile_90.tif",
            "probability_exceeding_1mm_h.tif",
            "accumulation_mean_60min.tif",
            "accumulation_percentile_90_60min.tif",
        ]
    ), names

    with rasterio.open(tmp_path / "tif" / "rain_rate_mean.tif") as src:
        assert src.count == 12 and src.shape == (8, 10)
        assert src.crs is not None and "Transverse_Mercator" in src.crs.to_wkt()
        assert src.transform.a == 1000.0 and src.transform.e == -1000.0
        assert src.transform.c == -600_000.0 and src.transform.f == 650_000.0
        band1 = src.read(1)
        np.testing.assert_allclose(band1, products["rain_rate_mean"].isel(forecast_step=0).values, rtol=1e-6)
        assert src.descriptions[0].startswith("2025-08-28T22:25")
        assert src.compression is not None

    with rasterio.open(tmp_path / "tif" / "accumulation_mean_60min.tif") as src:
        assert src.count == 1


def test_geotiff_needs_a_grid(tmp_path):
    pytest.importorskip("rasterio")
    from convgru_ensemble.geotiff import write_products_geotiffs

    _, forecast = _forecast(georeferenced=False)
    with pytest.raises(ValueError, match="x and y coordinates"):
        write_products_geotiffs(ensemble_products(forecast), tmp_path)
