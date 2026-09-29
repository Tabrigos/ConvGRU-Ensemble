from unittest.mock import MagicMock, patch

from huggingface_hub.errors import EntryNotFoundError

from convgru_ensemble.hub import from_pretrained


def _download_without_safetensors(repo_id, filename, revision=None):
    if filename in ("model.safetensors", "config.json"):
        raise EntryNotFoundError(f"{filename} not in {repo_id}")
    return "/tmp/cached/model.ckpt"


@patch("convgru_ensemble.hub.hf_hub_download", side_effect=_download_without_safetensors)
def test_from_pretrained_falls_back_to_the_checkpoint(mock_download):
    with patch("convgru_ensemble.lightning_model.RadarLightningModel") as mock_cls:
        mock_cls.from_checkpoint.return_value = MagicMock()
        from_pretrained("it4lia/irene", filename="model.ckpt", device="cpu", revision="abc123")

    mock_download.assert_any_call(repo_id="it4lia/irene", filename="model.ckpt", revision="abc123")
    mock_cls.from_checkpoint.assert_called_once_with("/tmp/cached/model.ckpt", device="cpu")


@patch("convgru_ensemble.hub.hf_hub_download", return_value="/tmp/cached/file")
def test_from_pretrained_prefers_safetensors(mock_download):
    with patch("convgru_ensemble.weights.load_weights") as mock_load:
        mock_load.return_value = MagicMock()
        model = from_pretrained("it4lia/irene", device="cpu")

    assert model is mock_load.return_value
    mock_load.assert_called_once_with("/tmp/cached/file", "/tmp/cached/file", device="cpu")
    filenames = [c.kwargs["filename"] for c in mock_download.call_args_list]
    assert filenames == ["model.safetensors", "config.json"]
