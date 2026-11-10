from pathlib import Path

import mlflow.pytorch
import pytest
import torch
from PIL import Image

from inference_backend.model_loader import LoadedModel


@pytest.fixture
def metadata():
    return {
        "image_size": [28, 28],
        "mean": 0.1307,
        "std": 0.3081,
        "preprocessing": "mnist-v1",
        "labels": list(range(10)),
    }


@pytest.fixture
def loaded_model(metadata):
    network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
    with torch.no_grad():
        network[1].weight.zero_()
        network[1].bias.copy_(torch.arange(10, dtype=torch.float32))
    network.eval()
    return LoadedModel(
        network=network, name="MNIST", version="7", metadata=metadata, device=torch.device("cpu")
    )


@pytest.fixture
def image_path(tmp_path: Path):
    path = tmp_path / "digit.png"
    Image.new("RGB", (56, 56), color="white").save(path)
    return path


@pytest.fixture
def packaged_model(tmp_path: Path, loaded_model, metadata):
    path = tmp_path / "packaged"
    scripted = torch.jit.trace(loaded_model.network, torch.zeros(1, 1, 28, 28))
    mlflow.pytorch.save_model(
        scripted,
        str(path),
        pip_requirements=["torch==2.6.0", "mlflow==3.18.0"],
        serialization_format="pickle",
        metadata=metadata,
    )
    return path
