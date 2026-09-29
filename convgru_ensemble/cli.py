"""Command-line interface for ConvGRU-Ensemble inference and serving."""

import time

import fire
import xarray as xr

from .horizon import check_forecast_steps, trained_forecast_steps
from .inputs import TRAINED_PAST_STEPS, select_past, to_rain_rate_mm_h
from .output import NETCDF_ENCODING, PHYSICAL_FLOOR_MM_H, apply_floor, build_forecast_dataset
from .products import DEFAULT_ACCUMULATIONS_MIN, DEFAULT_PERCENTILES, DEFAULT_THRESHOLDS_MM_H, ensemble_products


def _load_model(
    checkpoint: str | None = None,
    hub_repo: str | None = None,
    device: str = "cpu",
    hub_revision: str | None = None,
    weights: str | None = None,
):
    """Load the model from safetensors weights, a local checkpoint or HuggingFace Hub."""
    from .lightning_model import RadarLightningModel

    if weights is not None:
        print(f"Loading model from safetensors: {weights}")
        from .weights import load_weights

        return load_weights(weights, device=device)
    if hub_repo is not None:
        print(f"Loading model from HuggingFace Hub: {hub_repo}" + (f" @ {hub_revision}" if hub_revision else ""))
        return RadarLightningModel.from_pretrained(hub_repo, device=device, revision=hub_revision)
    if checkpoint is not None:
        print(f"Loading model from checkpoint: {checkpoint}")
        return RadarLightningModel.from_checkpoint(checkpoint, device=device)
    raise ValueError("One of --weights, --checkpoint or --hub-repo must be provided.")


def predict(
    input: str,
    checkpoint: str | None = None,
    hub_repo: str | None = None,
    hub_revision: str | None = None,
    weights: str | None = None,
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
        hub_revision: Git revision of the Hub repo (branch, tag or commit sha); pin a sha in production.
        weights: Path to model.safetensors (config.json next to it). Loads without unpickling anything.
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
    model = _load_model(checkpoint, hub_repo, device, hub_revision, weights)
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


def _numbers(value) -> tuple[float, ...]:
    """Parse '1,5,20' (or a fire-parsed list/tuple/number) into a tuple of floats; empty means none."""
    if value is None or value == "" or value == ():
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, str):
        return tuple(float(v) for v in value.split(",") if v.strip())
    return tuple(float(v) for v in value)


def products(
    input: str,
    output: str = "products.nc",
    geotiff_dir: str | None = None,
    thresholds: str | tuple = DEFAULT_THRESHOLDS_MM_H,
    percentiles: str | tuple = DEFAULT_PERCENTILES,
    accumulations: str | tuple = DEFAULT_ACCUMULATIONS_MIN,
):
    """
    Summarize an ensemble forecast into products: mean, median, spread, percentiles,
    probability of exceedance and accumulations, as NetCDF and optionally GeoTIFF.

    Args:
        input: Forecast NetCDF written by `predict` (or by the API).
        output: Path for the products NetCDF.
        geotiff_dir: If given, also write one GeoTIFF per product (needs the 'geo' extra).
        thresholds: Rain rates in mm/h for the probability of exceedance, e.g. '1,5,20'. Empty string disables.
        percentiles: Percentiles of the ensemble (0-100), e.g. '10,50,90'. Empty string disables.
        accumulations: Accumulation windows in minutes from the first step, e.g. '30,60'. Empty string disables.
    """
    print(f"Loading forecast: {input}")
    forecast = xr.open_dataset(input)
    t0 = time.perf_counter()
    result = ensemble_products(
        forecast,
        thresholds=_numbers(thresholds),
        percentiles=_numbers(percentiles),
        accumulations_min=_numbers(accumulations),
    )
    print(f"Products: {', '.join(v for v in result.data_vars if v != 'crs')} ({time.perf_counter() - t0:.2f}s)")

    encoding = {name: {"zlib": True, "complevel": 4} for name in result.data_vars if name != "crs"}
    result.to_netcdf(output, encoding=encoding)
    print(f"Products saved to: {output}")

    if geotiff_dir:
        from .geotiff import write_products_geotiffs

        paths = write_products_geotiffs(result, geotiff_dir)
        print(f"GeoTIFF: {len(paths)} files in {geotiff_dir}")


def export_weights(
    output: str,
    checkpoint: str | None = None,
    hub_repo: str | None = None,
    hub_revision: str | None = None,
):
    """
    Convert a Lightning checkpoint to model.safetensors + config.json.

    This is the one step that unpickles the checkpoint: run it once on a
    checkpoint you trust, then serve from the exported weights (--weights /
    MODEL_WEIGHTS), which load without executing anything.

    Args:
        output: Directory to write model.safetensors and config.json into.
        checkpoint: Path to a local .ckpt checkpoint file.
        hub_repo: HuggingFace Hub repo ID to download the checkpoint from. Alternative to --checkpoint.
        hub_revision: Git revision of the Hub repo (branch, tag or commit sha).
    """
    from huggingface_hub import hf_hub_download

    from .weights import export_checkpoint

    if checkpoint is None and hub_repo is None:
        raise ValueError("Either --checkpoint or --hub-repo must be provided.")
    if checkpoint is None:
        print(
            f"Downloading checkpoint from HuggingFace Hub: {hub_repo}" + (f" @ {hub_revision}" if hub_revision else "")
        )
        checkpoint = hf_hub_download(repo_id=hub_repo, filename="model.ckpt", revision=hub_revision)
    weights_path, config_path = export_checkpoint(checkpoint, output)
    print(f"Weights saved to: {weights_path}")
    print(f"Config saved to: {config_path}")


def serve(
    checkpoint: str | None = None,
    hub_repo: str | None = None,
    hub_revision: str | None = None,
    weights: str | None = None,
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
        hub_revision: Git revision of the Hub repo (branch, tag or commit sha); env HF_REVISION.
        weights: Path to model.safetensors (config.json next to it); env MODEL_WEIGHTS. Takes precedence.
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
    if hub_revision is not None:
        os.environ["HF_REVISION"] = hub_revision
    if weights is not None:
        os.environ["MODEL_WEIGHTS"] = weights
    os.environ.setdefault("DEVICE", device)
    if max_forecast_steps is not None:
        os.environ["MAX_FORECAST_STEPS"] = str(max_forecast_steps)

    import uvicorn

    uvicorn.run("convgru_ensemble.serve:app", host=host, port=port)


def main():
    fire.Fire({"predict": predict, "products": products, "serve": serve, "export-weights": export_weights})


if __name__ == "__main__":
    main()
