"""Weights in safetensors: a distribution format that is not a pickle.

A Lightning ``.ckpt`` is a pickle: loading it runs whatever code the file
contains, which is fine for the official checkpoint and unacceptable for a
file of unknown origin on a production server. This module exports the
weights and the inference hyperparameters to ``model.safetensors`` plus a
``config.json``, and loads them back without executing anything.
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

# Hyperparameters the model needs for inference. Optimizer and scheduler are
# training-only and are not rebuilt from a config file.
INFERENCE_HPARAMS = (
    "input_channels",
    "num_blocks",
    "ensemble_size",
    "noisy_decoder",
    "forecast_steps",
    "loss_class",
    "loss_params",
    "masked_loss",
)


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', str(value))}".strip(".")


def config_from_hparams(hparams: dict, training: dict | None = None) -> dict:
    """The JSON config written next to the weights: inference hyperparameters plus training notes."""
    config = {k: _jsonable(hparams[k]) for k in INFERENCE_HPARAMS if k in hparams}
    notes = {k: _jsonable(v) for k, v in hparams.items() if k not in INFERENCE_HPARAMS}
    if training:
        notes.update(training)
    if notes:
        config["training"] = notes
    return config


def export_weights(model, out_dir: str | Path, training: dict | None = None) -> tuple[Path, Path]:
    """
    Write ``model.safetensors`` and ``config.json`` for a loaded model.

    Parameters
    ----------
    model : RadarLightningModel
        The model whose weights and hyperparameters are exported.
    out_dir : str or Path
        Directory to write into, created if needed.
    training : dict or None, optional
        Notes about the training run (epoch, steps, source) kept in the config.

    Returns
    -------
    weights_path, config_path : tuple of Path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    config = config_from_hparams(dict(model.hparams), training)
    weights_path = out_dir / WEIGHTS_FILE
    config_path = out_dir / CONFIG_FILE
    save_file(state, str(weights_path), metadata={"format": "pt", "config": json.dumps(config)})
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return weights_path, config_path


def export_checkpoint(checkpoint_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """
    Convert a Lightning checkpoint to safetensors plus config.

    This is the one step that loads a pickle: run it once, on a checkpoint
    you trust, and distribute the result.
    """
    from .lightning_model import RadarLightningModel

    checkpoint_path = Path(checkpoint_path)
    model = RadarLightningModel.from_checkpoint(str(checkpoint_path), device="cpu")
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    training = {
        "source_checkpoint": checkpoint_path.name,
        "source_sha256": sha256_of(checkpoint_path),
        "epoch": raw.get("epoch"),
        "global_step": raw.get("global_step"),
        "pytorch_lightning_version": raw.get("pytorch-lightning_version"),
    }
    return export_weights(model, out_dir, training={k: v for k, v in training.items() if v is not None})


def load_weights(weights_path: str | Path, config_path: str | Path | None = None, device: str = "cpu"):
    """
    Build the model from ``config.json`` and load ``model.safetensors`` into it.

    Nothing is unpickled: the config is JSON and the weights are plain
    tensors. Missing or unexpected tensors are an error.
    """
    from .lightning_model import RadarLightningModel

    weights_path = Path(weights_path)
    config_path = Path(config_path) if config_path else weights_path.with_name(CONFIG_FILE)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hparams = {k: config[k] for k in INFERENCE_HPARAMS if k in config}
    for key in ("input_channels", "num_blocks"):
        if key not in hparams:
            raise ValueError(f"{config_path} lacks '{key}', needed to build the model")
    model = RadarLightningModel(**hparams)
    state = load_file(str(weights_path), device=device)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
