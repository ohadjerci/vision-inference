"""Resolve a registry selection once and load its immutable packaged artifact."""

from dataclasses import dataclass
from typing import Any, cast

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
    metadata: dict[str, Any]
    device: torch.device


def load_model(settings: Settings, device: str = "cpu") -> LoadedModel:
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    target = torch.device(device)
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
    network = cast(torch.nn.Module, mlflow.pytorch.load_model(path, map_location=target))
    network = network.to(target)
    network.eval()
    return LoadedModel(
        network=network,
        name=settings.model_name,
        version=version,
        metadata=metadata,
        device=target,
    )
