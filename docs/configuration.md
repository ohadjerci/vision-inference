# Configuration

Run commands from the **inference repository root**. Compose reads the root `.env`
and maps supported values into the container. The Python package itself reads
process environment variables; it does not load a `.env` file automatically.

## Environment variables

| Variable | Compose default | Meaning |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | HTTP(S) server for registry lookup and downloads. |
| `MODEL_NAME` | `MNIST` | Nonempty registered model name; `/` and `@` are rejected. |
| `MODEL_VERSION` | empty | Positive integer version; empty selects the `best_model` alias. |
| `MODEL_CACHE_DIR` | `/home/inference/.cache/models` | Writable download directory. |
| `LOCAL_UID`, `LOCAL_GID` | `1000:1000` | User for the lockfile maintenance service only. |

See [`.env.example`](../.env.example). Per-run Compose `-e` values override the
container environment. `settings.py` validates the server URL and version and
checks cache write access before loading a model.

For callers using the Python package outside Compose, the URI fallback is
`http://localhost:50001` and the cache fallback is
`~/.cache/vision-inference`. Containers use the different defaults above.

## Model selection

By default, each invocation resolves `MNIST@best_model` once, then downloads its
exact numeric version. The selected alias is fixed by the loader; it is not a
configurable environment value.

Select a specific version instead, replacing `1` with an existing version:

```bash
docker compose run --rm -e MODEL_VERSION=1 backend /inputs/digit.png
```

Missing aliases, unavailable versions, incompatible metadata, and network failures
are errors. The loader does not choose another version or fall back to a cached
model silently. The [architecture guide](architecture.md#selection-requirement-mismatch)
explains the training-side validation/test-loss mismatch.

## Device selection

The CLI defaults to CPU; use `--device cpu` to select it explicitly. For CUDA,
install the NVIDIA driver and NVIDIA Container Toolkit and expose the GPU:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --build --rm backend --device cuda /inputs/digit.png
```

The GPU image uses PyTorch with CUDA 12.4. A compatible driver is required.
CUDA selection reports an error when unavailable, without falling back to CPU.

## Input and cache paths

The host path `inputs/example.jpg` maps to `/inputs/example.jpg` in the container.
Pass the container path to the CLI. Images must be readable by the runtime's
non-root user (`inference`, UID 1000); the input mount is read-only.

The named cache volume is mounted at `/home/inference/.cache/models`. Changing
`MODEL_CACHE_DIR` does not move that mount: another path must be writable, and
its downloads are ephemeral unless you also provide a persistent mount.
For a fresh temporary download during troubleshooting:

```bash
docker compose run --rm -e MODEL_CACHE_DIR=/tmp/model-check \
  backend /inputs/digit.png
```

Keep `.env`, input images, caches, and generated outputs out of Git. Commit the
shareable example and dependency lockfile instead.
