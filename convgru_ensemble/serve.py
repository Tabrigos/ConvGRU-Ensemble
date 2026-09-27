"""FastAPI inference server for ConvGRU-Ensemble nowcasting model."""

import io
import os
import tempfile
import time
from contextlib import asynccontextmanager

import numpy as np
import xarray as xr
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .inputs import InputUnitsError, to_rain_rate_mm_h
from .output import NETCDF_ENCODING, build_forecast_dataset

_model = None

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
    checkpoint = os.environ.get("MODEL_CHECKPOINT")
    hub_repo = os.environ.get("HF_REPO_ID")

    if hub_repo:
        return RadarLightningModel.from_pretrained(hub_repo, device=device)
    elif checkpoint:
        return RadarLightningModel.from_checkpoint(checkpoint, device=device)
    else:
        raise RuntimeError("Set MODEL_CHECKPOINT or HF_REPO_ID environment variable.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = _load_model()
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
        "ensemble_size": hp.ensemble_size,
        "noisy_decoder": hp.noisy_decoder,
        "loss_class": str(hp.loss_class),
        "device": str(_model.device),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="NetCDF file with rain rate data (T, H, W)"),  # noqa: B008
    variable: str = Query("RR", description="Name of the rain rate variable"),  # noqa: B008
    units: str | None = Query(None, description="Input units, overriding the file's 'units' attribute"),  # noqa: B008
    forecast_steps: int = Query(12, ge=1, le=48, description="Number of future 5-min steps (max 48 = 4h)"),  # noqa: B008
    ensemble_size: int = Query(10, ge=1, le=10, description="Number of ensemble members (max 10)"),  # noqa: B008
):
    """
    Run ensemble nowcasting inference on uploaded NetCDF data.

    Accepts a NetCDF file containing past radar rain rate observations and
    returns NetCDF predictions with ensemble forecasts.
    """
    t0 = time.perf_counter()

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
        raise HTTPException(
            status_code=422,
            detail=f"Failed to read NetCDF file: {exc}",
        ) from exc

    # Check variable exists
    if variable not in ds:
        available = list(ds.data_vars)
        raise HTTPException(
            status_code=422,
            detail=f"Variable '{variable}' not found. Available: {available}",
        )

    da = ds[variable]

    # Must be 3D
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

    if da.shape[0] < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 2 timesteps, got {da.shape[0]}.",
        )

    try:
        rain = to_rain_rate_mm_h(da, units=units)
    except InputUnitsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if np.isinf(rain.values).any():
        raise HTTPException(
            status_code=422,
            detail="Input data contains Inf values.",
        )

    # Replace NaN with 0 (no rain) — common for masked radar pixels
    past = np.nan_to_num(rain.values, nan=0.0)

    # Run inference
    preds = _model.predict(past, forecast_steps=forecast_steps, ensemble_size=ensemble_size)

    elapsed = time.perf_counter() - t0

    # Build output NetCDF, carrying over coordinates, grid mapping and valid times
    ds_out = build_forecast_dataset(
        preds,
        da,
        ds,
        attrs={
            "elapsed_seconds": f"{elapsed:.3f}",
            "input_units": rain.units,
            "input_messages": "; ".join(rain.messages),
        },
    )

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    ds_out.to_netcdf(tmp_out_path, engine="netcdf4", encoding=NETCDF_ENCODING)
    with open(tmp_out_path, "rb") as fh:
        out_bytes = fh.read()
    os.unlink(tmp_out_path)

    return Response(
        content=out_bytes,
        media_type="application/x-netcdf",
        headers={
            "Content-Disposition": "attachment; filename=predictions.nc",
            "X-Elapsed-Seconds": f"{elapsed:.3f}",
            "X-Input-Units": rain.units,
            "X-Input-Messages": "; ".join(rain.messages),
        },
    )
