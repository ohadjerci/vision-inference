"""Exercise the API without a live registry."""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from inference_backend import api
from inference_backend.model_loader import LoadedModel
from inference_backend.predictor import Prediction


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(api.Settings, "from_env", lambda: object())
    monkeypatch.setattr(
        api,
        "load_model",
        Mock(return_value=LoadedModel(Mock(), "MNIST", "1", {}, torch.device("cpu"))),
    )
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_and_reuse_model(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def fake_predict(model: object, path: Path, metadata: object) -> Prediction:
        assert model is api.app.state.model
        assert path.exists()
        with Image.open(path) as image:
            assert image.format == "PNG"
        seen.append(path)
        return Prediction(7, 0.95, "1")

    monkeypatch.setattr(api, "predict", fake_predict)
    image = BytesIO()
    Image.new("L", (28, 28)).save(image, format="PNG")
    assert client.get("/health").json()["status"] == "ok"
    for _ in range(2):
        response = client.post(
            "/predict", files={"file": ("digit.png", image.getvalue(), "image/png")}
        )
        body = response.json()
        assert body.pop("latency_ms") >= 0
        assert body == {
            "prediction": 7,
            "confidence": 0.95,
            "model_version": "1",
            "device": "cpu",
        }
    assert api.load_model.call_count == 1  # type: ignore[attr-defined]
    assert all(not path.exists() for path in seen)


def test_invalid_image_and_missing_file(client: TestClient) -> None:
    response = client.post("/predict", files={"file": ("digit.png", b"invalid", "image/png")})
    assert response.status_code == 400
    assert client.post("/predict").status_code == 422


def test_prediction_failure_cleans_up(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[Path] = []

    def fail(model: object, path: Path, metadata: object) -> Prediction:
        paths.append(path)
        raise RuntimeError("model failed")

    monkeypatch.setattr(api, "predict", fail)
    response = client.post("/predict", files={"file": ("digit.png", b"data", "image/png")})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    assert paths and not paths[0].exists()


def test_main_starts_server(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock()
    monkeypatch.setattr(api.uvicorn, "run", run)
    api.main()
    run.assert_called_once_with(api.app, host="0.0.0.0", port=8000)


def test_device_selection(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api.torch.cuda, "synchronize", Mock())
    seen: list[str] = []

    def fake_predict(model: LoadedModel, path: Path, metadata: object) -> Prediction:
        seen.append(str(model.device))
        return Prediction(7, 0.95, "1")

    monkeypatch.setattr(api, "predict", fake_predict)
    network = api.app.state.model.network
    for device in ("cuda", "cuda", "cpu"):
        response = client.post(
            "/predict", data={"device": device}, files={"file": ("digit.png", b"image")}
        )
        assert response.status_code == 200
        assert response.json()["device"] == device
        assert response.json()["latency_ms"] >= 0
    assert seen == ["cuda", "cuda", "cpu"]
    assert network.to.call_count == 2
    assert api.load_model.call_count == 1  # type: ignore[attr-defined]


def test_unavailable_and_invalid_device(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: False)
    response = client.post(
        "/predict", data={"device": "cuda"}, files={"file": ("digit.png", b"image")}
    )
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
    response = client.post(
        "/predict", data={"device": "invalid"}, files={"file": ("digit.png", b"image")}
    )
    assert response.status_code == 422
