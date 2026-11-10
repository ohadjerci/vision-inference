"""HTTP interface to the reusable inference pipeline."""

import logging
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal

import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from PIL import UnidentifiedImageError

from inference_backend.model_loader import load_model
from inference_backend.predictor import predict
from inference_backend.settings import Settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = load_model(Settings.from_env())
    app.state.model_lock = Lock()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "cuda_available": torch.cuda.is_available()}


@app.post("/predict")
def predict_image(
    request: Request, file: UploadFile, device: Annotated[Literal["cpu", "cuda"], Form()] = "cpu"
) -> dict[str, object]:
    try:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "upload.png"
            with path.open("wb") as destination:
                shutil.copyfileobj(file.file, destination)
            with request.app.state.model_lock:
                model = request.app.state.model
                if device == "cuda" and not torch.cuda.is_available():
                    raise HTTPException(
                        status_code=503, detail="CUDA is unavailable on the backend"
                    )
                target = torch.device(device)
                if model.device != target:
                    model.network.to(target)
                    model = replace(model, device=target)
                    request.app.state.model = model
                if device == "cuda":
                    torch.cuda.synchronize()
                started = perf_counter()
                result = asdict(predict(model, path, model.metadata))
                if device == "cuda":
                    torch.cuda.synchronize()
                latency_ms = (perf_counter() - started) * 1000
                return {**result, "device": device, "latency_ms": latency_ms}
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail=f"Invalid image or prediction: {error}"
        ) from error
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from error
    finally:
        file.file.close()


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)
