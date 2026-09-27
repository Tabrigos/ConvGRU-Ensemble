import numpy as np
import pytest
import xarray as xr

from convgru_ensemble.output import DPC_RADAR_GRID_MAPPING, NETCDF_ENCODING, build_forecast_dataset, forecast_times


def _dpc_window(t=6, h=8, w=10, with_time=True):
    """A small window of the DPC 1 km grid, like a crop of the sample file."""
    y = 649_500.0 - 1000.0 * np.arange(h)
    x = -599_500.0 + 1000.0 * np.arange(w)
    coords = {"y": ("y", y, {"units": "m"}), "x": ("x", x, {"units": "m"})}
    if with_time:
        coords["time"] = ("time", np.datetime64("2025-08-28T21:55") + np.timedelta64(5, "m") * np.arange(t))
    da = xr.DataArray(np.zeros((t, h, w), dtype=np.float32), dims=["time", "y", "x"], coords=coords, name="RR")
    return xr.Dataset({"RR": da})


def test_dpc_grid_gets_coordinates_crs_and_times():
    ds = _dpc_window()
    preds = np.full((3, 12, 8, 10), 0.5, dtype=np.float32)
    out = build_forecast_dataset(preds, ds["RR"], ds)

    assert out["precipitation_forecast"].dims == ("ensemble_member", "forecast_step", "y", "x")
    np.testing.assert_array_equal(out["y"].values, ds["y"].values)
    np.testing.assert_array_equal(out["x"].values, ds["x"].values)
    assert out["precipitation_forecast"].attrs["grid_mapping"] == "crs"
    assert out["crs"].attrs["grid_mapping_name"] == "transverse_mercator"
    assert out["crs"].attrs["latitude_of_projection_origin"] == 42.0
    assert list(out["ensemble_member"].values) == [0, 1, 2]
    assert list(out["forecast_step"].values) == list(range(1, 13))
    expected = ds["time"].values[-1] + np.timedelta64(5, "m") * np.arange(1, 13)
    np.testing.assert_array_equal(out["forecast_time"].values, expected)
    assert out["forecast_reference_time"].values == ds["time"].values[-1]
    assert out.attrs["timestep_minutes"] == 5.0
    assert out.attrs["ensemble_size"] == 3 and out.attrs["forecast_steps"] == 12


def test_existing_grid_mapping_is_copied():
    ds = _dpc_window()
    ds["crs"] = xr.DataArray(np.int32(0), attrs={"grid_mapping_name": "latitude_longitude", "note": "custom"})
    ds["RR"].attrs["grid_mapping"] = "crs"
    out = build_forecast_dataset(np.zeros((1, 2, 8, 10), dtype=np.float32), ds["RR"], ds)
    assert out["crs"].attrs == {"grid_mapping_name": "latitude_longitude", "note": "custom"}


def test_input_without_coordinates_still_builds():
    ds = xr.Dataset({"RR": xr.DataArray(np.zeros((4, 8, 8), dtype=np.float32), dims=["time", "y", "x"])})
    out = build_forecast_dataset(np.zeros((2, 12, 8, 8), dtype=np.float32), ds["RR"], ds)
    assert out["precipitation_forecast"].shape == (2, 12, 8, 8)
    assert "crs" not in out
    assert "forecast_time" not in out.coords
    assert "grid_mapping" not in out["precipitation_forecast"].attrs


def test_non_dpc_grid_gets_no_crs():
    ds = _dpc_window()
    ds = ds.assign_coords(x=ds["x"] * 10)  # 10 km spacing: not the DPC grid
    out = build_forecast_dataset(np.zeros((1, 1, 8, 10), dtype=np.float32), ds["RR"], ds)
    assert "crs" not in out
    np.testing.assert_array_equal(out["x"].values, ds["x"].values)


def test_timestep_is_inferred_from_input():
    ds = _dpc_window(t=3)
    ds = ds.assign_coords(time=np.datetime64("2025-01-01T00:00") + np.timedelta64(10, "m") * np.arange(3))
    times, reference, step = forecast_times(ds["RR"], 4)
    assert step == np.timedelta64(10, "m")
    assert times[0] == reference + np.timedelta64(10, "m")


def test_shape_mismatch_is_rejected():
    ds = _dpc_window()
    with pytest.raises(ValueError):
        build_forecast_dataset(np.zeros((1, 1, 4, 4), dtype=np.float32), ds["RR"], ds)


def test_netcdf_roundtrip_keeps_georeference(tmp_path):
    ds = _dpc_window()
    out = build_forecast_dataset(np.random.default_rng(0).random((2, 3, 8, 10), dtype=np.float32), ds["RR"], ds)
    path = tmp_path / "pred.nc"
    out.to_netcdf(path, encoding=NETCDF_ENCODING)
    back = xr.open_dataset(path)
    np.testing.assert_array_equal(back["x"].values, ds["x"].values)
    np.testing.assert_array_equal(back["forecast_time"].values, out["forecast_time"].values)
    assert back["crs"].attrs["crs_wkt"] == DPC_RADAR_GRID_MAPPING["crs_wkt"]
    assert back["precipitation_forecast"].encoding.get("zlib") is True
