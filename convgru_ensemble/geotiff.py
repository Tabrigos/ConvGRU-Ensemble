"""Write forecast products as GeoTIFF files for GIS tools and web maps.

Requires the ``geo`` extra (``rasterio``). One file per product, with one
band per forecast step (or per accumulation window); products with an
extra dimension such as the threshold or the percentile give one file per
value. Files are tiled, compressed and carry overviews, so they open fast
in QGIS and can be served as Cloud-Optimized GeoTIFFs.
"""

from itertools import product
from pathlib import Path

import numpy as np
import xarray as xr


def _rasterio():
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("GeoTIFF export needs the 'geo' extra: uv sync --extra geo") from exc
    return rasterio, from_origin


def grid_transform(ds: xr.Dataset):
    """
    Affine transform and CRS of the dataset grid.

    The transform comes from the ``x`` and ``y`` coordinates (cell centres,
    regular spacing); the CRS from the ``crs_wkt`` attribute of the grid
    mapping variable when present.
    """
    rasterio, from_origin = _rasterio()
    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError("GeoTIFF export needs x and y coordinates on the grid")
    x = ds["x"].values.astype(float)
    y = ds["y"].values.astype(float)
    if x.size < 2 or y.size < 2:
        raise ValueError("GeoTIFF export needs at least a 2x2 grid")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if not (np.allclose(np.diff(x), dx) and np.allclose(np.diff(y), dy)):
        raise ValueError("GeoTIFF export needs a regular grid")
    if dx <= 0:
        raise ValueError("x must increase; flip the grid before exporting")
    # A north-up raster has y decreasing along rows; a y-increasing grid is flipped on write.
    top = y[0] - dy / 2 if dy < 0 else y[-1] + dy / 2
    transform = from_origin(x[0] - dx / 2, top, dx, abs(dy))
    crs = None
    if "crs" in ds and "crs_wkt" in ds["crs"].attrs:
        crs = rasterio.crs.CRS.from_wkt(ds["crs"].attrs["crs_wkt"])
    return transform, crs, dy > 0


def _band_stack(da: xr.DataArray, flip: bool) -> np.ndarray:
    """Bands first, then rows (north to south), then columns."""
    if da.ndim == 2:
        data = da.values[np.newaxis]
    elif da.ndim == 3:
        data = da.values
    else:
        raise ValueError(f"Cannot write {da.ndim}D variable '{da.name}' as bands")
    if flip:
        data = data[:, ::-1, :]
    return np.ascontiguousarray(data, dtype=np.float32)


def write_geotiff(da: xr.DataArray, path: Path, transform, crs, flip: bool, descriptions=None) -> Path:
    """Write one raster with the bands of ``da`` (its leading non-spatial dim, if any)."""
    rasterio, _ = _rasterio()
    data = _band_stack(da, flip)
    nodata = -1.0
    data = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "predictor": 3,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(**{k: str(v) for k, v in da.attrs.items()})
        if descriptions:
            for i, text in enumerate(descriptions, start=1):
                dst.set_band_description(i, str(text))
        factors = [f for f in (2, 4, 8, 16) if min(data.shape[1:]) // f >= 64]
        if factors:
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
    return path


def _band_labels(da: xr.DataArray) -> list[str] | None:
    if da.ndim < 3:
        return None
    dim = da.dims[0]
    if dim == "forecast_step" and "forecast_time" in da.coords:
        return [np.datetime_as_string(t, unit="m") for t in da["forecast_time"].values]
    return [f"{dim}={v}" for v in da[dim].values]


def write_products_geotiffs(products: xr.Dataset, out_dir: Path, prefix: str = "") -> list[Path]:
    """
    Write every product of :func:`convgru_ensemble.products.ensemble_products` as GeoTIFF.

    Products with a leading ``threshold``, ``percentile`` or ``accumulation``
    dimension are split into one file per value, e.g.
    ``probability_exceeding_5mm_h.tif`` or ``rain_rate_percentile_90.tif``,
    with one band per forecast step. Returns the written paths.
    """
    transform, crs, flip = grid_transform(products)
    out_dir = Path(out_dir)
    written = []
    for name, da in products.data_vars.items():
        if name == "crs" or not {"y", "x"} <= set(da.dims):
            continue
        extra = [d for d in da.dims if d not in ("y", "x", "forecast_step")]
        # One file per combination of the extra dimensions (threshold, percentile, accumulation...)
        for values in product(*(da[d].values for d in extra)):
            sub = da.sel(dict(zip(extra, values, strict=True)))
            labels = [_value_label(d, v) for d, v in zip(extra, values, strict=True)]
            path = out_dir / ("_".join([f"{prefix}{name}", *labels]) + ".tif")
            written.append(write_geotiff(sub, path, transform, crs, flip, _band_labels(sub)))
    return written


def _value_label(dim: str, value) -> str:
    text = f"{float(value):g}".replace(".", "p")
    if dim == "threshold":
        return f"{text}mm_h"
    if dim == "accumulation":
        return f"{text}min"
    return text
