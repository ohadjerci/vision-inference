"""Metadata-driven image preprocessing and inference."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from inference_backend.model_loader import LoadedModel


@dataclass(frozen=True)
class Prediction:
    prediction: int
    confidence: float
    model_version: str


def _dimensions(metadata: dict[str, Any], key: str, length: int) -> list[int]:
    value = metadata.get(key)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError(f"Metadata {key} must contain {length} positive integers")
    return list(value)


def _normalization(metadata: dict[str, Any], key: str, channels: int) -> torch.Tensor:
    value = metadata.get(key)
    values = value if isinstance(value, (list, tuple)) else [value]
    if len(values) not in (1, channels) or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        or (key == "std" and item <= 0)
        for item in values
    ):
        raise ValueError(
            f"Metadata {key} must contain finite values for each channel"
            + (" with positive standard deviations" if key == "std" else "")
        )
    return torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)


def preprocess(image_path: Path, metadata: dict[str, Any]) -> torch.Tensor:
    channels, height, width = _dimensions(metadata, "input_shape", 3)
    if _dimensions(metadata, "image_size", 2) != [height, width]:
        raise ValueError("Metadata image_size must match input_shape height and width")
    if channels not in (1, 3):
        raise ValueError("Metadata input_shape must specify 1 or 3 input channels")
    mean = _normalization(metadata, "mean", channels)
    std = _normalization(metadata, "std", channels)
    with Image.open(image_path) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Expected a PNG or JPEG image")
        resized = image.convert("L" if channels == 1 else "RGB").resize(
            (width, height), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype=np.float32).copy() / 255.0
    if channels == 1:
        pixels = pixels[:, :, np.newaxis]
    inputs = torch.from_numpy(pixels).permute(2, 0, 1)
    return ((inputs - mean) / std).unsqueeze(0)


def predict(model: LoadedModel, image_path: Path, metadata: dict[str, Any]) -> Prediction:
    classes = metadata.get("num_classes")
    if type(classes) is not int or classes < 1:
        raise ValueError("Metadata num_classes must be a positive integer")
    labels = metadata.get("labels")
    if labels is not None and (not isinstance(labels, (list, tuple)) or len(labels) != classes):
        raise ValueError("Metadata labels must match num_classes")
    inputs = preprocess(image_path, metadata).to(model.device)
    model.network.eval()
    with torch.inference_mode():
        logits = model.network(inputs)
        if not isinstance(logits, torch.Tensor) or logits.shape != (1, classes):
            raise ValueError(f"Model must return one vector of {classes} class logits")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Model returned non-finite logits")
        probabilities = torch.softmax(logits, dim=1)
        confidence, label = probabilities.max(dim=1)
    return Prediction(int(label.item()), float(confidence.item()), model.version)
