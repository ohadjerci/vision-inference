import json
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

import inference_backend.cli as cli
import inference_backend.predictor as predictor
from inference_backend.model_loader import LoadedModel
from inference_backend.predictor import predict, preprocess


def test_normalization_shape_and_polarity(image_path, metadata):
    inputs = preprocess(image_path, metadata)
    assert inputs.shape == (1, 1, 28, 28)
    assert inputs.dtype == torch.float32
    torch.testing.assert_close(inputs, torch.full_like(inputs, (1.0 - 0.1307) / 0.3081))


def test_prediction_confidence_and_version(loaded_model, image_path):
    result = predict(loaded_model, image_path, loaded_model.metadata)
    assert result.prediction == 9
    assert result.model_version == "7"
    assert result.confidence == pytest.approx(torch.softmax(torch.arange(10.0), dim=0)[9].item())


def test_black_input_and_jpeg(tmp_path, metadata):
    path = tmp_path / "black.jpg"
    Image.new("L", (28, 28), color=0).save(path)
    torch.testing.assert_close(
        preprocess(path, metadata), torch.full((1, 1, 28, 28), -0.1307 / 0.3081)
    )


def test_invalid_image(tmp_path, loaded_model):
    path = tmp_path / "broken.png"
    path.write_bytes(b"invalid")
    with pytest.raises(OSError):
        predict(loaded_model, path, loaded_model.metadata)


@pytest.mark.parametrize("width", [9, 11])
def test_invalid_logits(image_path, width, metadata):
    network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, width))
    with pytest.raises(ValueError, match="10 class logits"):
        predict(
            LoadedModel(network, "MNIST", "1", metadata, torch.device("cpu")), image_path, metadata
        )


def test_cli_json_and_error(loaded_model, image_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_model", lambda settings, device: loaded_model)
    assert cli.main([str(image_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["prediction"] == 9
    assert output["model_version"] == "7"
    assert 0 <= output["confidence"] <= 1
    assert cli.main([str(image_path.parent / "missing.png")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Image does not exist" in captured.err


def test_rgb_non_square_and_per_channel_normalization(image_path, metadata):
    metadata.update(image_size=[4, 7], mean=[0.0, 0.5, 1.0], std=[1.0, 0.5, 0.25])
    result = preprocess(image_path, metadata)
    assert result.shape == (1, 3, 4, 7)
    torch.testing.assert_close(result[0, :, 0, 0], torch.tensor([1.0, 1.0, 0.0]))


@pytest.mark.parametrize(
    "key,value",
    [
        ("input_shape", None),
        ("input_shape", [2, 28, 28]),
        ("input_shape", [1, 0, 28]),
        ("image_size", [0, 27]),
        ("mean", None),
        ("mean", [float("nan")]),
        ("mean", [1, 2, 3, 4]),
        ("std", None),
        ("std", [0]),
        ("std", [-1]),
        ("std", [float("inf")]),
    ],
)
def test_invalid_preprocessing_metadata(image_path, metadata, key, value):
    metadata[key] = value
    with pytest.raises(ValueError):
        preprocess(image_path, metadata)


def test_optional_input_shape_must_match_size(image_path, metadata):
    metadata["input_shape"] = [1, 27, 28]
    with pytest.raises(ValueError, match="image_size"):
        preprocess(image_path, metadata)


@pytest.mark.parametrize(
    "key,value",
    [
        ("labels", None),
        ("labels", []),
        ("labels", "digits"),
    ],
)
def test_invalid_output_metadata(loaded_model, image_path, metadata, key, value):
    metadata[key] = value
    with pytest.raises(ValueError, match=key):
        predict(loaded_model, image_path, metadata)


def test_dynamic_classes_and_inference_mode(image_path, metadata):
    metadata.update(labels=["a", "b"])

    class Network(torch.nn.Module):
        def forward(self, inputs):
            assert torch.is_inference_mode_enabled()
            assert not self.training
            return torch.tensor([[0.0, 1.0]], device=inputs.device)

    model = LoadedModel(Network(), "model", "1", metadata, torch.device("cpu"))
    assert predict(model, image_path, metadata).prediction == 1


def test_predict_moves_inputs_to_selected_device(monkeypatch, image_path, metadata):
    inputs = Mock()
    inputs.to.return_value = torch.zeros(1)
    monkeypatch.setattr(predictor, "preprocess", lambda path, data: inputs)
    network = Mock(return_value=torch.zeros(1, 10))
    model = LoadedModel(network, "model", "1", metadata, torch.device("cuda"))
    predict(model, image_path, metadata)
    inputs.to.assert_called_once_with(torch.device("cuda"))
    network.eval.assert_called_once()


def test_rejects_nonfinite_logits(loaded_model, image_path):
    loaded_model.network[1].bias.data[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        predict(loaded_model, image_path, loaded_model.metadata)


def test_cli_passes_cuda(loaded_model, image_path, monkeypatch, capsys):
    load = Mock(return_value=loaded_model)
    monkeypatch.setattr(cli, "load_model", load)
    assert cli.main(["--device", "cuda", str(image_path)]) == 0
    assert load.call_args.args[1] == "cuda"
    assert json.loads(capsys.readouterr().out)["model_version"] == "7"
