"""MNIST preprocessing, inference, and postprocessing reusable by a future API."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference_backend.model_loader import LoadedModel


@dataclass(frozen=True)
class Prediction:
    prediction: int
    confidence: float
    model_version: str


def preprocess(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Expected a PNG or JPEG image")
        grayscale = image.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
        pixels = np.asarray(grayscale, dtype=np.float32).copy() / 255.0
    return (torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0) - 0.1307) / 0.3081


def predict(model: LoadedModel, image_path: Path) -> Prediction:
    inputs = preprocess(image_path)
    with torch.inference_mode():
        logits = model.network(inputs)
        if not isinstance(logits, torch.Tensor) or logits.shape != (1, 10):
            raise ValueError("Model must return one vector of 10 digit logits")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Model returned non-finite logits")
        probabilities = torch.softmax(logits, dim=1)
        confidence, label = probabilities.max(dim=1)
    return Prediction(int(label.item()), float(confidence.item()), model.version)
