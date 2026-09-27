import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from convgru_ensemble.lightning_model import RadarLightningModel


@pytest.fixture
def mock_model():
    model = MagicMock(spec=RadarLightningModel)
    model.hparams = MagicMock()
    model.hparams.input_channels = 1
    model.hparams.num_blocks = 2
    model.hparams.forecast_steps = 12
    model.hparams.ensemble_size = 2
    model.hparams.noisy_decoder = False
    model.hparams.loss_class = "crps"
    model.device = "cpu"
    model.predict.return_value = np.zeros((10, 12, 8, 8), dtype=np.float32)
    return model


@pytest.fixture
def client(mock_model):
    from fastapi.testclient import TestClient

    with patch("convgru_ensemble.serve._load_model", return_value=mock_model):
        from convgru_ensemble.serve import app

        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["architecture"] == "ConvGRU-Ensemble EncoderDecoder"
    assert data["num_blocks"] == 2


def test_predict_returns_netcdf(client):
    # Create a small NetCDF file in memory
    ds = xr.Dataset({"RR": xr.DataArray(np.zeros((9, 8, 8), dtype=np.float32), dims=["time", "y", "x"])})
    buf = io.BytesIO()
    ds.to_netcdf(buf, engine="scipy")
    buf.seek(0)

    resp = client.post(
        "/predict?forecast_steps=12&ensemble_size=10",
        files={"file": ("input.nc", buf, "application/x-netcdf")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/x-netcdf"
    assert "X-Elapsed-Seconds" in resp.headers


def test_predict_accepts_netcdf4(client, tmp_path):
    # NetCDF4 is an HDF5 file: it must be recognized without libmagic
    ds = xr.Dataset({"RR": xr.DataArray(np.zeros((9, 8, 8), dtype=np.float32), dims=["time", "y", "x"])})
    path = tmp_path / "input.nc"
    ds.to_netcdf(path, engine="netcdf4")
    with open(path, "rb") as fh:
        resp = client.post("/predict?forecast_steps=12&ensemble_size=10", files={"file": ("input.nc", fh)})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/x-netcdf"


def test_predict_rejects_non_netcdf(client):
    resp = client.post("/predict", files={"file": ("input.nc", io.BytesIO(b"not a netcdf file at all"))})
    assert resp.status_code == 422
    assert "Expected a NetCDF file" in resp.json()["detail"]


def test_netcdf_engine_for_signatures():
    from convgru_ensemble.serve import netcdf_engine_for

    assert netcdf_engine_for(b"\x89HDF\r\n\x1a\n" + b"\0" * 8) == "h5netcdf"
    assert netcdf_engine_for(b"CDF\x01" + b"\0" * 8) == "scipy"
    assert netcdf_engine_for(b"CDF\x02" + b"\0" * 8) == "scipy"
    assert netcdf_engine_for(b"CDF\x05" + b"\0" * 8) == "scipy"
    assert netcdf_engine_for(b"PK\x03\x04") is None
    assert netcdf_engine_for(b"") is None


def _upload(frames):
    ds = xr.Dataset({"RR": xr.DataArray(np.zeros((frames, 8, 8), dtype=np.float32), dims=["time", "y", "x"])})
    buf = io.BytesIO()
    ds.to_netcdf(buf, engine="scipy")
    buf.seek(0)
    return {"file": ("input.nc", buf, "application/x-netcdf")}


def test_predict_uses_last_six_frames_by_default(client, mock_model):
    resp = client.post("/predict?forecast_steps=12&ensemble_size=10", files=_upload(9))
    assert resp.status_code == 200, resp.text
    past = mock_model.predict.call_args.args[0]
    assert past.shape == (6, 8, 8)
    assert "last 6 of 9" in resp.headers["X-Input-Messages"]


def test_predict_past_steps_can_be_changed(client, mock_model):
    resp = client.post("/predict?forecast_steps=12&ensemble_size=10&past_steps=4", files=_upload(9))
    assert resp.status_code == 200, resp.text
    assert mock_model.predict.call_args.args[0].shape == (4, 8, 8)


def test_predict_rejects_too_few_frames(client):
    resp = client.post("/predict?forecast_steps=12&ensemble_size=10", files=_upload(4))
    assert resp.status_code == 422
    assert "Need at least 6 timesteps, got 4" in resp.json()["detail"]
