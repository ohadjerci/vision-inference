"""Console interface; loading and prediction remain usable independently."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from inference_backend.model_loader import load_model
from inference_backend.predictor import predict
from inference_backend.settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict an MNIST-style digit using MLflow")
    parser.add_argument(
        "image", type=Path, help="PNG/JPEG path (light digit on a dark background)"
    )
    args = parser.parse_args(argv)
    try:
        if not args.image.is_file():
            raise FileNotFoundError(f"Image does not exist: {args.image}")
        model = load_model(Settings.from_env())
        result = predict(model, args.image)
    except Exception as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(result), allow_nan=False))
    return 0
