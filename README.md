# Vision inference

Standalone CPU inference for the MNIST model published by the separate
[training repository](https://github.com/ohadjerci/ml-engineering-lab).
The installed Python package has no dependency on that checkout. The interface
is a CLI; `frontend/` is reserved for later and there is no web API.

## Start the shared MLflow server

In the training repository, run:

```bash
mkdir -p logs/train logs/eval logs/mlflow/db logs/mlflow/artifacts
docker compose up --build -d mlflow
docker compose run --build --rm train
```

The server owns the `vision-mlflow` Docker network. Tracking metadata and the
registry use `logs/mlflow/db/mlflow.db`; model files use
`logs/mlflow/artifacts/`. Only that server needs these paths. Host clients use
`http://localhost:50001`; containers on the shared network use
`http://mlflow:5000`. Inside a container, localhost refers to that container.
Do not stop the shared server while inference is downloading a model.

Training selects the lowest-validation-loss checkpoint, tests it, and registers
a self-contained TorchScript model named `MNIST`. Standalone evaluation can also
publish its evaluated checkpoint. The `best_model` alias moves only when
`test/loss` strictly improves; ties retain the existing alias. Publishers must run
sequentially. Inference resolves that alias once to a numeric version and loads
that exact version, even if the alias subsequently moves.

## Run inference

Docker Compose and a running shared server are required. No GPU is needed here.

```bash
cp .env.example .env
mkdir -p inputs
# Place a PNG/JPEG digit image in inputs/digit.png.
docker compose build backend
docker compose run --rm backend /inputs/digit.png
# Select an explicit version instead of best_model:
docker compose run --rm -e MODEL_VERSION=1 backend /inputs/digit.png
```

Output is JSON: `{"prediction": 7, "confidence": 0.98, "model_version": "1"}`.
Confidence is the predicted class softmax probability, not a calibrated score.
Use a light digit on a dark background. Images are converted to grayscale,
resized to 28×28, and normalized using the training mean/std. There is no automatic
inversion, cropping, or centering.

Configuration is validated from environment variables:

| Variable | Meaning | Compose default |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | Shared HTTP(S) tracking and registry URI | `http://mlflow:5000` |
| `MODEL_NAME` | Registered model name | `MNIST` |
| `MODEL_VERSION` | Positive numeric version; empty uses `best_model` | empty |
| `MODEL_CACHE_DIR` | Writable download directory | `/home/inference/.cache/models` |

The cache lives in a Docker named volume. The only host mount is the read-only
`inputs/` directory. No training checkout or training logs are mounted. Missing
models, incompatible metadata, invalid images, and connection failures return a
nonzero exit status with an error on stderr. Only load artifacts from a trusted
MLflow registry.

## Backend development and checks

`backend/` owns the package, dependency lockfile, tests, and Docker build.
`settings.py`, `model_loader.py`, and `predictor.py` can be reused independently of
the CLI. The Docker image installs the package non-editably; no `sys.path`
modifications are used.

```bash
docker compose run --build --rm tests
docker compose run --rm tests uv run --locked ruff check .
docker compose run --rm tests uv run --locked ruff format --check .
docker compose run --rm tests uv run --locked mypy
docker compose run --rm tests uv lock --check
docker compose run --rm tests uv pip check
```

These checks do not require the shared network or server. For dependency updates:

```bash
LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" docker compose run --build --rm lock uv lock
```

Commit `backend/uv.lock`. Never commit `.env`, model downloads, inputs, or generated
outputs. GitHub Actions installs the locked backend and runs tests, Ruff, and
MyPy on CPU. Live integration verification requires the separate training server
and a published model, and is performed locally through Docker.

## Historical training logs

The training server was upgraded from a file backend. The original file history
and a verified timestamped backup remain under the training repository's
`logs/mlflow/`. Supported metadata migration to SQLite retains original artifact
locations; it does not convert them to HTTP-proxied URIs. New runs use the
`minst_project_http` experiment and proxied artifacts. Historical local artifacts
are not standalone inference inputs; publish a new packaged registry version.
