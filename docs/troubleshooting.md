# Troubleshooting

Run commands from the **inference repository root** unless another repository is
explicitly named.

## Network vision-mlflow not found

Start the existing MLflow service from the **training repository root**:

```bash
docker compose up --build -d mlflow
```

The training project creates the shared network. Backend tests and quality checks
run without it. See [shared connectivity](mlflow.md).

## Connection refused or invalid host header

Use `http://mlflow:5000` inside the inference container, not `localhost:50001`.
In the **training repository**, check:

```bash
docker compose ps mlflow
docker compose logs --tail 50 mlflow
```

If you choose another hostname, it must be reachable and accepted by the server's
allowed-host configuration. The default published port is host-loopback only.

## Model or best_model alias missing

Check the `MNIST` registered model in MLflow. A tracked experiment/run is not
itself a registered model. Training must finish testing and publication, or a
standalone evaluation must publish a compatible artifact. Select an existing
numeric version with `MODEL_VERSION` if it has no alias yet. The application does
not create a replacement model automatically.

## Image does not exist or prediction is poor

A host file `inputs/digit.png` must be passed as `/inputs/digit.png`. Use a readable
PNG/JPEG containing one light digit on a dark background. The implemented
preprocessing does not invert, crop, or center it. A smoke-trained model is not
expected to recognize digits reliably; softmax confidence is not calibrated accuracy.

## Cache is not writable

The runtime uses UID 1000. Keep the default named volume or provide a directory
writable by that user. A custom `MODEL_CACHE_DIR` does not automatically create a
new persistent mount. See [input and cache paths](configuration.md#input-and-cache-paths).

## TorchScript or CUDA-versus-CPU warning

Training and inference use PyTorch 2.6.0 with different CUDA/CPU wheel suffixes.
MLflow can warn about that difference and TorchScript dispatch. A warning alone
is not a failed prediction; check the exit status and JSON output. Do not hide an
actual load error or bypass the packaged preprocessing metadata validation.

## Historical local model artifacts

The training server's file-store history was retained and its metadata migrated
to SQLite. Migration did not rewrite local artifact paths into HTTP-proxied URIs.
A legacy run may therefore be visible in the UI while its artifact cannot be
loaded from this independent container.

Publish a new compatible model through the shared server. Do not mount the training
checkout or logs here as a workaround. The training project's `docs/migration.md`
records the historical procedure and limitations. The current default training
experiment is `project_http`; old experiment names may remain in its database.
