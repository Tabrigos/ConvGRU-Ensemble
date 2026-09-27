"""Command-line interface for ConvGRU-Ensemble inference and serving."""

import time

import fire
import xarray as xr

from .horizon import check_forecast_steps, trained_forecast_steps
from .inputs import TRAINED_PAST_STEPS, select_past, to_rain_rate_mm_h
from .output import NETCDF_ENCODING, PHYSICAL_FLOOR_MM_H, apply_floor, build_forecast_dataset


def _load_model(checkpoint: str | None = None, hub_repo: str | None = None, device: str = "cpu"):
    """Load model from local checkpoint or HuggingFace Hub."""
    from .lightning_model import RadarLightningModel

    if hub_repo is not None:
        print(f"Loading model from HuggingFace Hub: {hub_repo}")
        return RadarLightningModel.from_pretrained(hub_repo, device=device)
    elif checkpoint is not None:
        print(f"Loading model from checkpoint: {checkpoint}")
        return RadarLightningModel.from_checkpoint(checkpoint, device=device)
    else:
        raise ValueError("Either --checkpoint or --hub-repo must be provided.")


def predict(
    input: str,
    checkpoint: str | None = None,
    hub_repo: str | None = None,
    variable: str = "RR",
    units: str | None = None,
    past_steps: int = TRAINED_PAST_STEPS,
    forecast_steps: int = 12,
    max_forecast_steps: int | None = None,
    ensemble_size: int = 10,
    min_rain_rate: float = PHYSICAL_FLOOR_MM_H,
    device: str = "cpu",
    output: str = "predictions.nc",
):
    """
    Run inference on a NetCDF input file and save predictions as NetCDF.

    Args:
        input: Path to input NetCDF file with rain rate data (T, H, W) or (T, Y, X).
        checkpoint: Path to local .ckpt checkpoint file.
        hub_repo: HuggingFace Hub repo ID (e.g., 'it4lia/irene'). Alternative to --checkpoint.
        variable: Name of the rain rate variable in the NetCDF file.
        units: Units of the input, overriding the file's 'units' attribute (e.g. 'mm/h', 'kg m-2 s-1').
        past_steps: Number of past frames given to the model, taken from the end of the file (default 6, as in training).
        forecast_steps: Number of future timesteps to forecast (default 12, the trained horizon).
        max_forecast_steps: Cap on forecast_steps. Default is the trained horizon; raise it to extrapolate beyond.
        ensemble_size: Number of ensemble members to generate.
        min_rain_rate: Values at or below this (mm/h) are written as 0. Default: the model's physical floor (~0.036); 0 disables.
        device: Device for inference ('cpu' or 'cuda').
        output: Path for the output NetCDF file.
    """
    model = _load_model(checkpoint, hub_repo, device)
    trained = trained_forecast_steps(model)
    horizon_warning = check_forecast_steps(forecast_steps, trained, max_forecast_steps)
    if horizon_warning:
        print(f"Warning: {horizon_warning}")

    # Load input data
    print(f"Loading input: {input}")
    ds = xr.open_dataset(input)
    if variable not in ds:
        available = list(ds.data_vars)
        raise ValueError(f"Variable '{variable}' not found. Available: {available}")

    da = ds[variable]  # (T, H, W) or similar
    if da.ndim != 3:
        raise ValueError(f"Expected 3D data (T, H, W), got shape {da.shape}")

    da, dropped = select_past(da, past_steps)
    if dropped:
        print(f"Input: {dropped}")
    rain = to_rain_rate_mm_h(da, units=units)
    for message in rain.messages:
        print(f"Input: {message}")
    print(f"Input shape: {rain.values.shape}, units: {rain.units}")
    past = rain.values

    # Run inference
    t0 = time.perf_counter()
    preds = model.predict(past, forecast_steps=forecast_steps, ensemble_size=ensemble_size)
    elapsed = time.perf_counter() - t0
    preds = apply_floor(preds, min_rain_rate)
    print(f"Output shape: {preds.shape} (ensemble, time, H, W)")
    print(f"Elapsed: {elapsed:.2f}s")

    # Build output dataset, carrying over coordinates, grid mapping and valid times
    extra = {
        "source_file": str(input),
        "past_steps": past_steps,
        "trained_forecast_steps": trained,
        "min_rain_rate": min_rain_rate,
    }
    if horizon_warning:
        extra["beyond_training_horizon"] = horizon_warning
    ds_out = build_forecast_dataset(preds, da, ds, attrs=extra)

    ds_out.to_netcdf(output, encoding=NETCDF_ENCODING)
    print(f"Predictions saved to: {output}")


def serve(
    checkpoint: str | None = None,
    hub_repo: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cpu",
    max_forecast_steps: int | None = None,
):
    """
    Start the FastAPI inference server.

    Args:
        checkpoint: Path to local .ckpt checkpoint file.
        hub_repo: HuggingFace Hub repo ID (e.g., 'it4lia/irene'). Alternative to --checkpoint.
        host: Host to bind to.
        port: Port to listen on.
        device: Device for inference ('cpu' or 'cuda').
        max_forecast_steps: Cap on forecast_steps per request (env MAX_FORECAST_STEPS). Default: the trained horizon.
    """
    import os

    if checkpoint is not None:
        os.environ["MODEL_CHECKPOINT"] = checkpoint
    if hub_repo is not None:
        os.environ["HF_REPO_ID"] = hub_repo
    os.environ.setdefault("DEVICE", device)
    if max_forecast_steps is not None:
        os.environ["MAX_FORECAST_STEPS"] = str(max_forecast_steps)

    import uvicorn

    uvicorn.run("convgru_ensemble.serve:app", host=host, port=port)


def main():
    fire.Fire({"predict": predict, "serve": serve})


if __name__ == "__main__":
    main()
