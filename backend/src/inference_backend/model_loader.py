"""Resolve a registry selection once and load its immutable packaged artifact."""

from dataclasses import dataclass
from typing import cast

import mlflow.artifacts
import mlflow.pytorch
import torch
from mlflow import MlflowClient
from mlflow.models import Model

from inference_backend.settings import Settings


@dataclass(frozen=True)
class LoadedModel:
    network: torch.nn.Module
    name: str
    version: str


def load_model(settings: Settings) -> LoadedModel:
    client = MlflowClient(tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri)
    if settings.model_version is None:
        selected = client.get_model_version_by_alias(settings.model_name, "best_model")
    else:
        selected = client.get_model_version(settings.model_name, settings.model_version)
    version = str(selected.version)
    if not version.isdigit() or int(version) < 1:
        raise ValueError("Registry returned an invalid model version")
    # Use the numeric URI after alias resolution so alias changes cannot change this load.
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{settings.model_name}/{version}",
        dst_path=str(settings.cache_dir / settings.model_name / version),
        tracking_uri=settings.tracking_uri,
        registry_uri=settings.tracking_uri,
    )
    metadata = Model.load(path).metadata or {}
    expected = {
        "preprocessing": "mnist-v1",
        "image_size": [28, 28],
        "mean": 0.1307,
        "std": 0.3081,
        "labels": list(range(10)),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Model preprocessing metadata is incompatible with mnist-v1")
    network = cast(torch.nn.Module, mlflow.pytorch.load_model(path, map_location="cpu"))
    network.eval()
    return LoadedModel(network=network, name=settings.model_name, version=version)
