"""FastAPI inference server for ConvGRU-Ensemble nowcasting model."""

import io
import os
import tempfile
import time
from contextlib import asynccontextmanager

import numpy as np
import xarray as xr
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .horizon import HorizonError, check_forecast_steps, trained_forecast_steps
from .inputs import TRAINED_PAST_STEPS, InputError, select_past, to_rain_rate_mm_h
from .output import PHYSICAL_FLOOR_MM_H, apply_floor, build_forecast_dataset
from .products import DEFAULT_ACCUMULATIONS_MIN, DEFAULT_PERCENTILES, DEFAULT_THRESHOLDS_MM_H, ensemble_products

_model = None
_max_forecast_steps: int | None = None

# File signatures: NetCDF classic and 64-bit offset ("CDF\x01", "CDF\x02"), CDF-5 ("CDF\x05"),
# and NetCDF4, which is an HDF5 file. Checked directly, so no libmagic is needed.
_CLASSIC_SIGNATURES = (b"CDF\x01", b"CDF\x02", b"CDF\x05")
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def netcdf_engine_for(content: bytes) -> str | None:
    """Return the xarray engine able to read ``content``, or ``None`` if it is not NetCDF."""
    if content[:8] == _HDF5_SIGNATURE:
        return "h5netcdf"
    if content[:4] in _CLASSIC_SIGNATURES:
        return "scipy"
    return None


def _load_model():
    from .lightning_model import RadarLightningModel

    device = os.environ.get("DEVICE", "cpu")
    weights = os.environ.get("MODEL_WEIGHTS")
    checkpoint = os.environ.get("MODEL_CHECKPOINT")
    hub_repo = os.environ.get("HF_REPO_ID")
    revision = os.environ.get("HF_REVISION") or None

    if weights:
        from .weights import load_weights

        return load_weights(weights, device=device)
    if hub_repo:
        return RadarLightningModel.from_pretrained(hub_repo, device=device, revision=revision)
    if checkpoint:
        return RadarLightningModel.from_checkpoint(checkpoint, device=device)
    raise RuntimeError("Set MODEL_WEIGHTS, MODEL_CHECKPOINT or HF_REPO_ID environment variable.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _max_forecast_steps
    _model = _load_model()
    cap = os.environ.get("MAX_FORECAST_STEPS")
    _max_forecast_steps = int(cap) if cap else None
    yield
    _model = None


app = FastAPI(
    title="ConvGRU-Ensemble Nowcasting API",
    version="0.1.0",
    description="Ensemble precipitation nowcasting from radar data",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    if _model is None:
        return {"error": "Model not loaded"}
    hp = _model.hparams
    return {
        "architecture": "ConvGRU-Ensemble EncoderDecoder",
        "input_channels": hp.input_channels,
        "num_blocks": hp.num_blocks,
        "forecast_steps": hp.forecast_steps,
        "max_forecast_steps": _max_forecast_steps or trained_forecast_steps(_model),
        "ensemble_size": hp.ensemble_size,
        "noisy_decoder": hp.noisy_decoder,
        "loss_class": str(hp.loss_class),
        "device": str(_model.device),
    }


def forecast_params(
    variable: str = Query("RR", description="Name of the rain rate variable"),  # noqa: B008
    units: str | None = Query(None, description="Input units, overriding the file's 'units' attribute"),  # noqa: B008
    past_steps: int = Query(  # noqa: B008
        TRAINED_PAST_STEPS, ge=1, le=48, description="Past frames given to the model, taken from the end of the file"
    ),
    forecast_steps: int = Query(  # noqa: B008
        12,
        ge=1,
        description="Number of future 5-min steps; capped at the trained horizon unless MAX_FORECAST_STEPS is set",
    ),
    ensemble_size: int = Query(10, ge=1, le=10, description="Number of ensemble members (max 10)"),  # noqa: B008
    min_rain_rate: float = Query(  # noqa: B008
        PHYSICAL_FLOOR_MM_H, ge=0, description="Values at or below this (mm/h) are returned as 0; 0 disables"
    ),
) -> dict:
    """Query parameters shared by the forecast endpoints."""
    return {
        "variable": variable,
        "units": units,
        "past_steps": past_steps,
        "forecast_steps": forecast_steps,
        "ensemble_size": ensemble_size,
        "min_rain_rate": min_rain_rate,
    }


def _numbers(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    """Parse a comma-separated list of numbers; None keeps the default, an empty string means none."""
    if text is None:
        return default
    try:
        return tuple(float(v) for v in text.split(",") if v.strip())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Expected comma-separated numbers, got '{text}'.") from exc


async def _forecast_from_upload(
    file: UploadFile,
    variable: str,
    units: str | None,
    past_steps: int,
    forecast_steps: int,
    ensemble_size: int,
    min_rain_rate: float,
) -> tuple[xr.Dataset, list[str], str]:
    """
    Validate the upload, run the model and build the georeferenced forecast dataset.

    Returns the dataset, the input messages and the units used. Every
    rejection is an HTTPException with a 4xx status and a plain reason.
    """
    t0 = time.perf_counter()

    trained = trained_forecast_steps(_model)
    try:
        horizon_warning = check_forecast_steps(forecast_steps, trained, _max_forecast_steps)
    except HorizonError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Read file and check size (max 100 MB)
    max_size = 100 * 1024 * 1024
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1024 / 1024:.0f} MB). Maximum is 100 MB.",
        )

    engine = netcdf_engine_for(content)
    if engine is None:
        raise HTTPException(
            status_code=422,
            detail="Expected a NetCDF file (classic 'CDF' or NetCDF4/HDF5 signature not found).",
        )
    try:
        ds = xr.open_dataset(io.BytesIO(content), engine=engine)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to read NetCDF file: {exc}") from exc

    if variable not in ds:
        available = list(ds.data_vars)
        raise HTTPException(status_code=422, detail=f"Variable '{variable}' not found. Available: {available}")

    da = ds[variable]
    if da.ndim != 3:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 3D variable (time, y, x), got {da.ndim}D with dims {da.dims}.",
        )

    # First dimension must be temporal
    time_names = {"time", "t", "step", "forecast_time", "lead_time"}
    first_dim = da.dims[0].lower()
    if first_dim not in time_names and da.shape[0] >= da.shape[1]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"First dimension should be time, got dims {da.dims} with shape {da.shape}. "
                "Expected shape (T, H, W) where T < H and T < W."
            ),
        )

    try:
        da, dropped = select_past(da, past_steps)
        rain = to_rain_rate_mm_h(da, units=units)
    except InputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if dropped:
        rain.messages.insert(0, dropped)
    if horizon_warning:
        rain.messages.append(horizon_warning)
    if np.isinf(rain.values).any():
        raise HTTPException(status_code=422, detail="Input data contains Inf values.")

    # Replace NaN with 0 (no rain) — common for masked radar pixels
    past = np.nan_to_num(rain.values, nan=0.0)

    preds = _model.predict(past, forecast_steps=forecast_steps, ensemble_size=ensemble_size)
    preds = apply_floor(preds, min_rain_rate)

    elapsed = time.perf_counter() - t0
    ds_out = build_forecast_dataset(
        preds,
        da,
        ds,
        attrs={
            "elapsed_seconds": f"{elapsed:.3f}",
            "past_steps": past_steps,
            "trained_forecast_steps": trained,
            "min_rain_rate": min_rain_rate,
            **({"beyond_training_horizon": horizon_warning} if horizon_warning else {}),
            "input_units": rain.units,
            "input_messages": "; ".join(rain.messages),
        },
    )
    return ds_out, rain.messages, rain.units


def _netcdf_response(ds: xr.Dataset, filename: str, headers: dict[str, str]) -> Response:
    """Serialize a dataset to a compressed NetCDF4 file and wrap it in a download response."""
    encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars if name != "crs"}
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    ds.to_netcdf(tmp_out_path, engine="netcdf4", encoding=encoding)
    with open(tmp_out_path, "rb") as fh:
        out_bytes = fh.read()
    os.unlink(tmp_out_path)
    return Response(
        content=out_bytes,
        media_type="application/x-netcdf",
        headers={"Content-Disposition": f"attachment; filename={filename}", **headers},
    )


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="NetCDF file with rain rate data (T, H, W)"),  # noqa: B008
    params: dict = Depends(forecast_params),  # noqa: B008
):
    """
    Run ensemble nowcasting inference on uploaded NetCDF data.

    Accepts a NetCDF file containing past radar rain rate observations and
    returns a NetCDF file with every ensemble member, on the input grid.
    """
    t0 = time.perf_counter()
    ds_out, messages, units = await _forecast_from_upload(file, **params)
    elapsed = time.perf_counter() - t0
    return _netcdf_response(
        ds_out,
        "predictions.nc",
        {
            "X-Elapsed-Seconds": f"{elapsed:.3f}",
            "X-Input-Units": units,
            "X-Input-Messages": "; ".join(messages),
        },
    )


@app.post("/products")
async def products(
    file: UploadFile = File(..., description="NetCDF file with rain rate data (T, H, W)"),  # noqa: B008
    params: dict = Depends(forecast_params),  # noqa: B008
    thresholds: str | None = Query(  # noqa: B008
        None,
        description=f"Rain rates in mm/h for the probability of exceedance, e.g. '1,5,20' (default {DEFAULT_THRESHOLDS_MM_H})",
    ),
    percentiles: str | None = Query(  # noqa: B008
        None, description=f"Ensemble percentiles 0-100, e.g. '10,50,90' (default {DEFAULT_PERCENTILES})"
    ),
    accumulations: str | None = Query(  # noqa: B008
        None, description=f"Accumulation windows in minutes, e.g. '30,60' (default {DEFAULT_ACCUMULATIONS_MIN})"
    ),
):
    """
    Run the ensemble and return its summary products instead of the members.

    Same input and parameters as ``/predict``; the answer is a NetCDF file
    with the ensemble mean, median and spread, the percentiles, the
    probability of exceeding each threshold and the accumulations, on the
    input grid with its georeference.
    """
    t0 = time.perf_counter()
    thresholds_v = _numbers(thresholds, DEFAULT_THRESHOLDS_MM_H)
    percentiles_v = _numbers(percentiles, DEFAULT_PERCENTILES)
    accumulations_v = _numbers(accumulations, DEFAULT_ACCUMULATIONS_MIN)
    if any(not 0 <= p <= 100 for p in percentiles_v):
        raise HTTPException(status_code=422, detail="Percentiles must be within 0-100.")

    ds_out, messages, units = await _forecast_from_upload(file, **params)
    summary = ensemble_products(
        ds_out, thresholds=thresholds_v, percentiles=percentiles_v, accumulations_min=accumulations_v
    )
    for key in ("past_steps", "min_rain_rate", "beyond_training_horizon", "input_units", "input_messages"):
        if key in ds_out.attrs:
            summary.attrs[key] = ds_out.attrs[key]
    elapsed = time.perf_counter() - t0
    summary.attrs["elapsed_seconds"] = f"{elapsed:.3f}"
    return _netcdf_response(
        summary,
        "products.nc",
        {
            "X-Elapsed-Seconds": f"{elapsed:.3f}",
            "X-Input-Units": units,
            "X-Input-Messages": "; ".join(messages),
        },
    )
