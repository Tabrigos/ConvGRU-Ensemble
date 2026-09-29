import json

import numpy as np
import pytest
import torch

from convgru_ensemble.lightning_model import RadarLightningModel
from convgru_ensemble.weights import CONFIG_FILE, WEIGHTS_FILE, export_checkpoint, export_weights, load_weights


def _small_model():
    torch.manual_seed(0)
    return RadarLightningModel(
        input_channels=1,
        num_blocks=2,
        ensemble_size=3,
        noisy_decoder=True,
        forecast_steps=4,
        loss_class="crps",
        loss_params={"temporal_lambda": 0.01},
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-4},
    )


def test_export_and_load_round_trip(tmp_path):
    model = _small_model()
    weights_path, config_path = export_weights(model, tmp_path, training={"epoch": 3})
    assert weights_path.name == WEIGHTS_FILE and config_path.name == CONFIG_FILE

    config = json.loads(config_path.read_text())
    assert config["num_blocks"] == 2 and config["ensemble_size"] == 3 and config["noisy_decoder"] is True
    assert config["training"]["epoch"] == 3
    assert config["training"]["optimizer_class"] == "torch.optim.adam.Adam"  # informational only

    loaded = load_weights(weights_path)
    assert loaded.hparams.num_blocks == 2 and loaded.hparams.forecast_steps == 4
    assert loaded.hparams.optimizer_class is None
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)

    past = np.random.default_rng(0).random((6, 32, 32), dtype=np.float32) * 5
    torch.manual_seed(1)
    a = model.predict(past, forecast_steps=2, ensemble_size=1)
    torch.manual_seed(1)
    b = loaded.predict(past, forecast_steps=2, ensemble_size=1)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_export_from_lightning_checkpoint(tmp_path):
    model = _small_model()
    ckpt = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "epoch": 7,
            "global_step": 1234,
            "pytorch-lightning_version": "2.6.0",
        },
        ckpt,
    )
    weights_path, config_path = export_checkpoint(ckpt, tmp_path / "out")
    config = json.loads(config_path.read_text())
    assert config["training"]["epoch"] == 7 and config["training"]["global_step"] == 1234
    assert config["training"]["source_checkpoint"] == "model.ckpt"
    assert len(config["training"]["source_sha256"]) == 64
    loaded = load_weights(weights_path)
    torch.testing.assert_close(
        loaded.state_dict()["model.encoder.blocks.0.convgru.cell.out_gate.weight"],
        model.state_dict()["model.encoder.blocks.0.convgru.cell.out_gate.weight"],
    )


def test_load_rejects_mismatched_weights(tmp_path):
    weights_path, config_path = export_weights(_small_model(), tmp_path)
    config = json.loads(config_path.read_text())
    config["num_blocks"] = 3  # weights are for 2 blocks
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError):
        load_weights(weights_path)


def test_load_needs_the_architecture(tmp_path):
    weights_path, config_path = export_weights(_small_model(), tmp_path)
    config_path.write_text(json.dumps({"ensemble_size": 3}))
    with pytest.raises(ValueError, match="input_channels"):
        load_weights(weights_path)
