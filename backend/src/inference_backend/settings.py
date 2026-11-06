"""Validated environment settings without command-line dependencies."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    tracking_uri: str
    model_name: str = "MNIST"
    model_version: str | None = None
    cache_dir: Path = Path.home() / ".cache" / "vision-inference"

    def __post_init__(self) -> None:
        url = urlsplit(self.tracking_uri)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) server URL")
        if not self.model_name.strip() or "/" in self.model_name or "@" in self.model_name:
            raise ValueError("MODEL_NAME must be a nonempty registry name without / or @")
        if self.model_version is not None:
            if not self.model_version.isascii() or not self.model_version.isdigit():
                raise ValueError("MODEL_VERSION must be a positive integer or empty")
            if int(self.model_version) < 1:
                raise ValueError("MODEL_VERSION must be positive")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Opening a temporary file checks actual write access, including container mounts.
        from tempfile import TemporaryFile

        with TemporaryFile(dir=self.cache_dir):
            pass

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        version = values.get("MODEL_VERSION", "").strip() or None
        return cls(
            tracking_uri=values.get("MLFLOW_TRACKING_URI", "http://localhost:50001"),
            model_name=values.get("MODEL_NAME", "MNIST"),
            model_version=version,
            cache_dir=Path(values.get("MODEL_CACHE_DIR", str(cls.cache_dir))).expanduser(),
        )
