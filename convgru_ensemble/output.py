"""Build georeferenced output datasets from model predictions.

The model works on plain arrays; this module attaches to the forecasts the
spatial coordinates, the grid mapping (CRS) and the valid times so that the
resulting NetCDF opens in the right place in GIS tools.
"""

import numpy as np
import xarray as xr

# Grid mapping of the Italian DPC radar mosaic (IT-DPC-SRI dataset):
# Transverse Mercator on WGS 84, origin 42N / 12.5E, scale factor 1,
# no false easting/northing, 1 km cells, x in [-600, 600] km, y in [-750, 650] km.
DPC_RADAR_GRID_MAPPING = {
    "grid_mapping_name": "transverse_mercator",
    "latitude_of_projection_origin": 42.0,
    "longitude_of_central_meridian": 12.5,
    "scale_factor_at_central_meridian": 1.0,
    "false_easting": 0.0,
    "false_northing": 0.0,
    "longitude_of_prime_meridian": 0.0,
    "semi_major_axis": 6378137.0,
    "inverse_flattening": 298.257223563,
    "reference_ellipsoid_name": "WGS 84",
    "horizontal_datum_name": "WGS84",
    "prime_meridian_name": "Greenwich",
    "geographic_crs_name": "WGS 84",
    "projected_crs_name": "DPC radar Transverse Mercator",
    "crs_wkt": (
        'PROJCS["DPC radar Transverse Mercator",'
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
        'PROJECTION["Transverse_Mercator"],'
        'PARAMETER["latitude_of_origin",42],PARAMETER["central_meridian",12.5],'
        'PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
        'UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    ),
    "proj4_params": "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
}

_DPC_X_RANGE = (-600_000.0, 600_000.0)
_DPC_Y_RANGE = (-750_000.0, 650_000.0)
_DPC_CELL = 1000.0

DEFAULT_TIMESTEP = np.timedelta64(5, "m")


def _spatial_dims(source: xr.DataArray) -> tuple[str, str]:
    return source.dims[-2], source.dims[-1]


def _looks_like_dpc_grid(source: xr.DataArray) -> bool:
    """Whether the spatial coordinates are (a window of) the DPC 1 km grid."""
    ydim, xdim = _spatial_dims(source)
    if ydim not in source.coords or xdim not in source.coords:
        return False
    y = source.coords[ydim].values
    x = source.coords[xdim].values
    if not (np.issubdtype(y.dtype, np.number) and np.issubdtype(x.dtype, np.number)):
        return False
    if y.size < 2 or x.size < 2:
        return False
    if not (np.allclose(np.abs(np.diff(y)), _DPC_CELL) and np.allclose(np.abs(np.diff(x)), _DPC_CELL)):
        return False
    return bool(
        _DPC_X_RANGE[0] <= x.min()
        and x.max() <= _DPC_X_RANGE[1]
        and _DPC_Y_RANGE[0] <= y.min()
        and y.max() <= _DPC_Y_RANGE[1]
    )


def grid_mapping_for(source: xr.DataArray, dataset: xr.Dataset | None = None) -> dict | None:
    """
    Return the CF grid mapping attributes for the source field, or ``None``.

    The mapping is taken from the variable named by the ``grid_mapping``
    attribute when the input dataset contains it; otherwise the DPC radar
    mapping is used when the coordinates match that grid.
    """
    name = source.attrs.get("grid_mapping")
    if name and dataset is not None and name in dataset.variables:
        return dict(dataset[name].attrs)
    if _looks_like_dpc_grid(source):
        return dict(DPC_RADAR_GRID_MAPPING)
    return None


def forecast_times(
    source: xr.DataArray, forecast_steps: int
) -> tuple[np.ndarray, np.datetime64, np.timedelta64] | None:
    """
    Valid times of the forecast steps, from the time coordinate of the input.

    Returns ``(times, reference_time, step)`` or ``None`` when the input has
    no datetime coordinate on its first dimension. The step is inferred from
    the last two observations; with a single observation the model's native
    5-minute step is assumed.
    """
    tdim = source.dims[0]
    if tdim not in source.coords:
        return None
    t = source.coords[tdim].values
    if not np.issubdtype(t.dtype, np.datetime64):
        return None
    t = t.astype("datetime64[ns]")
    step = (t[-1] - t[-2]) if t.size >= 2 else DEFAULT_TIMESTEP
    reference = t[-1]
    times = reference + step * np.arange(1, forecast_steps + 1)
    return times, reference, step


def build_forecast_dataset(
    preds: np.ndarray,
    source: xr.DataArray,
    dataset: xr.Dataset | None = None,
    attrs: dict | None = None,
) -> xr.Dataset:
    """
    Wrap model predictions in a georeferenced dataset.

    Parameters
    ----------
    preds : np.ndarray
        Forecast rain rate of shape ``(ensemble_member, forecast_step, H, W)``.
    source : xr.DataArray
        The input field ``(T, H, W)`` the forecast was computed from; its
        spatial coordinates, grid mapping and time coordinate are reused.
    dataset : xr.Dataset or None, optional
        The dataset the source comes from, used to look up the grid mapping
        variable. Default is ``None``.
    attrs : dict or None, optional
        Extra global attributes. Default is ``None``.

    Returns
    -------
    ds : xr.Dataset
        Dataset with ``precipitation_forecast`` and, when available, the
        spatial coordinates, a ``crs`` variable, ``forecast_time`` and
        ``forecast_reference_time``.
    """
    if preds.ndim != 4:
        raise ValueError(f"Expected predictions of shape (E, T, H, W), got {preds.shape}")
    ensemble_size, forecast_steps, height, width = preds.shape
    ydim, xdim = _spatial_dims(source)
    if source.sizes[ydim] != height or source.sizes[xdim] != width:
        raise ValueError(
            f"Prediction grid {height}x{width} does not match input grid {source.sizes[ydim]}x{source.sizes[xdim]}"
        )

    coords = {
        "ensemble_member": ("ensemble_member", np.arange(ensemble_size), {"long_name": "Ensemble member index"}),
        "forecast_step": (
            "forecast_step",
            np.arange(1, forecast_steps + 1),
            {"long_name": "Forecast step, counted from the last observation"},
        ),
    }
    # Spatial coordinates: everything in the input that lives on the spatial dims only.
    for name, coord in source.coords.items():
        if coord.dims and set(coord.dims) <= {ydim, xdim}:
            coords[name] = coord

    var_attrs = {
        "units": "mm h-1",
        "long_name": "Ensemble precipitation forecast",
        "standard_name": "rainfall_rate",
    }
    mapping = grid_mapping_for(source, dataset)
    if mapping is not None:
        var_attrs["grid_mapping"] = "crs"

    data_vars = {
        "precipitation_forecast": xr.DataArray(
            data=preds, dims=["ensemble_member", "forecast_step", ydim, xdim], attrs=var_attrs
        ),
    }
    if mapping is not None:
        data_vars["crs"] = xr.DataArray(np.int32(0), attrs=mapping)

    ds = xr.Dataset(data_vars, coords=coords)

    times = forecast_times(source, forecast_steps)
    if times is not None:
        valid, reference, step = times
        ds = ds.assign_coords(
            forecast_time=(
                "forecast_step",
                valid,
                {"standard_name": "time", "long_name": "Valid time of the forecast"},
            ),
            forecast_reference_time=(
                (),
                reference,
                {"standard_name": "forecast_reference_time", "long_name": "Time of the last observation"},
            ),
        )
        ds.attrs["timestep_minutes"] = float(step / np.timedelta64(1, "m"))

    ds.attrs.update(
        {
            "model": "ConvGRU-Ensemble",
            "forecast_steps": forecast_steps,
            "ensemble_size": ensemble_size,
            "Conventions": "CF-1.8",
        }
    )
    if attrs:
        ds.attrs.update(attrs)
    return ds


NETCDF_ENCODING = {"precipitation_forecast": {"zlib": True, "complevel": 4}}
