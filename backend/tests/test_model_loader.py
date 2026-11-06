import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.models import Model

import inference_backend.model_loader as loader
from inference_backend.settings import Settings


@pytest.mark.parametrize("requested", [None, "3"])
def test_resolves_then_downloads_exact_version(tmp_path, packaged_model, monkeypatch, requested):
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
    client.get_model_version.return_value = SimpleNamespace(version="3")
    monkeypatch.setattr(loader, "MlflowClient", Mock(return_value=client))
    download = Mock(return_value=str(packaged_model))
    monkeypatch.setattr(loader.mlflow.artifacts, "download_artifacts", download)
    settings = Settings(
        "http://mlflow:5000", model_version=requested, cache_dir=tmp_path / "cache"
    )
    loaded = loader.load_model(settings)
    expected = requested or "7"
    assert loaded.version == expected
    assert not loaded.network.training
    assert download.call_args.kwargs["artifact_uri"] == f"models:/MNIST/{expected}"
    assert download.call_args.kwargs["tracking_uri"] == settings.tracking_uri
    if requested is None:
        client.get_model_version_by_alias.assert_called_once_with("MNIST", "best_model")
    else:
        client.get_model_version_by_alias.assert_not_called()
        client.get_model_version.assert_called_once_with("MNIST", "3")


@pytest.mark.parametrize("requested", [None, "99"])
def test_registry_failure_has_no_fallback(tmp_path, monkeypatch, requested):
    client = Mock()
    client.get_model_version_by_alias.side_effect = RuntimeError("model missing")
    client.get_model_version.side_effect = RuntimeError("model missing")
    monkeypatch.setattr(loader, "MlflowClient", Mock(return_value=client))
    download = Mock()
    monkeypatch.setattr(loader.mlflow.artifacts, "download_artifacts", download)
    with pytest.raises(RuntimeError, match="model missing"):
        loader.load_model(
            Settings("http://mlflow:5000", model_version=requested, cache_dir=tmp_path)
        )
    download.assert_not_called()


@pytest.mark.parametrize("corrupt", [False, True])
def test_rejects_incompatible_or_corrupt_package(tmp_path, packaged_model, monkeypatch, corrupt):
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7")
    monkeypatch.setattr(loader, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(
        loader.mlflow.artifacts, "download_artifacts", lambda **kwargs: str(packaged_model)
    )
    if corrupt:
        for path in (packaged_model / "data").rglob("*.pth"):
            path.write_bytes(b"not a model")
    else:
        descriptor = Model.load(str(packaged_model))
        descriptor.metadata = {"preprocessing": "unsupported"}
        descriptor.save(str(packaged_model / "MLmodel"))
    with pytest.raises((RuntimeError, pickle.UnpicklingError) if corrupt else ValueError):
        loader.load_model(Settings("http://mlflow:5000", cache_dir=tmp_path / "cache"))


@pytest.mark.parametrize("value", ["file:///tmp/mlruns", "localhost:5000", ""])
def test_rejects_non_http_tracking_uri(tmp_path, value):
    with pytest.raises(ValueError, match="HTTP"):
        Settings(value, cache_dir=tmp_path)


@pytest.mark.parametrize("value", ["0", "-1", "latest", "1.5", "../1"])
def test_rejects_invalid_version(tmp_path, value):
    with pytest.raises(ValueError, match="MODEL_VERSION"):
        Settings("http://mlflow:5000", model_version=value, cache_dir=tmp_path)


def test_environment_configuration(tmp_path):
    settings = Settings.from_env(
        {
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MODEL_NAME": "MNIST",
            "MODEL_VERSION": " 3 ",
            "MODEL_CACHE_DIR": str(tmp_path / "cache"),
        }
    )
    assert settings.model_version == "3"
    assert settings.cache_dir.is_dir()
