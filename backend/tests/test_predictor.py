import json

import pytest
import torch
from PIL import Image

import inference_backend.cli as cli
from inference_backend.model_loader import LoadedModel
from inference_backend.predictor import predict, preprocess


def test_normalization_shape_and_polarity(image_path):
    inputs = preprocess(image_path)
    assert inputs.shape == (1, 1, 28, 28)
    assert inputs.dtype == torch.float32
    torch.testing.assert_close(inputs, torch.full_like(inputs, (1.0 - 0.1307) / 0.3081))


def test_prediction_confidence_and_version(loaded_model, image_path):
    result = predict(loaded_model, image_path)
    assert result.prediction == 9
    assert result.model_version == "7"
    assert result.confidence == pytest.approx(torch.softmax(torch.arange(10.0), dim=0)[9].item())


def test_black_input_and_jpeg(tmp_path):
    path = tmp_path / "black.jpg"
    Image.new("L", (28, 28), color=0).save(path)
    torch.testing.assert_close(preprocess(path), torch.full((1, 1, 28, 28), -0.1307 / 0.3081))


def test_invalid_image(tmp_path, loaded_model):
    path = tmp_path / "broken.png"
    path.write_bytes(b"invalid")
    with pytest.raises(OSError):
        predict(loaded_model, path)


@pytest.mark.parametrize("width", [9, 11])
def test_invalid_logits(tmp_path, image_path, width):
    network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, width))
    with pytest.raises(ValueError, match="10 digit logits"):
        predict(LoadedModel(network, "MNIST", "1"), image_path)


def test_cli_json_and_error(loaded_model, image_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_model", lambda settings: loaded_model)
    assert cli.main([str(image_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["prediction"] == 9
    assert output["model_version"] == "7"
    assert 0 <= output["confidence"] <= 1
    assert cli.main([str(image_path.parent / "missing.png")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Image does not exist" in captured.err
