# Vision Inference

## Project overview

Run standalone image predictions on CPU or NVIDIA GPU using a model from the shared MLflow
registry. **MNIST digit recognition is the first supported model.** The installed
Python backend downloads a packaged model without importing the training checkout.

[ML Engineering Lab](https://github.com/ohadjerci/ml-engineering-lab) trains,
evaluates, and registers models. This project provides the `vision-infer` CLI;
there is no inference web API or frontend yet.

## Prerequisites

- Linux x86-64, Git access to this repository, Docker Engine, and Docker Compose.
- The training project's MLflow server running on the same Docker host, with the
  shared `vision-mlflow` network already created.
- A compatible registered `MNIST` version, with a `best_model` alias for the
  default command. Follow the training project's quick start first.
- A PNG or JPEG digit image. CPU inference needs no GPU, host Python, or host uv.
- For CUDA: a compatible NVIDIA GPU and driver, plus the NVIDIA Container Toolkit.
- [just](https://github.com/casey/just) is optional and provides shortcuts for Docker commands.

See [shared-server setup](docs/mlflow.md) if the server, network, or model is missing.

## Quick start

Run commands in the **inference repository root** after cloning.

1. **Get the project.**

   ```bash
   git clone https://github.com/ohadjerci/vision-inference.git
   cd vision-inference
   ```

2. **Create local configuration.** Keep an existing `.env` if already configured.

   ```bash
   cp .env.example .env
   ```

   The example uses `http://mlflow:5000`, model `MNIST`, and the `best_model` alias.
   A numeric version can be selected [explicitly](docs/configuration.md#model-selection).

3. **Place an image in the input directory.** Replace the source path below with
   your image's real path. The filename inside the container will be `/inputs/digit.png`.

   ```bash
   mkdir -p inputs
   cp /absolute/path/to/your/digit.png inputs/digit.png
   ```

4. **Build and run.**

   ```bash
   docker compose build backend
   docker compose run --rm backend /inputs/digit.png
   ```

   CPU is the default. To use CUDA, expose the GPU with the optional Compose file:

   ```bash
   docker compose -f compose.yaml -f compose.gpu.yaml run --build --rm backend --device cuda /inputs/digit.png
   ```

   CUDA selection fails if no compatible GPU is available; it does not fall back to CPU.

   With `just` installed, use `just run /inputs/digit.png cpu` or
   `just run /inputs/digit.png gpu`. The device defaults to CPU when omitted.
   Run `just` to list the available recipes.

## Debugging in VS Code

Install the VS Code Python and Python Debugger extensions, then open the repository
root in VS Code. The debug container uses the same MLflow network and model settings
as the runtime container.

1. Run `just debug-up cpu` or `just debug-up gpu` to build and start the persistent
   container. It remains running between debug sessions. Use `just bash` for a shell.
2. Set breakpoints in `backend/src`, then run `just debug /inputs/digit.png cpu` or
   `just debug /inputs/digit.png gpu` with the same device in a terminal. Both
   commands default to CPU. The command waits for a debugger to connect.
3. In VS Code, start **Attach to Docker inference** from Run and Debug. The CLI
   executes after attaching, and its output appears in the terminal running `just`.
4. Run `just debug-down` when finished. Run `just debug` again for another image;
   source edits are available in the container without rebuilding.

Use `just build cpu` or `just build gpu` to build a runtime image explicitly;
the default is CPU. `just bash` and `just debug-down` operate on the active
debug container regardless of its device.

The debugger listens on port 5678, published only on the host's loopback interface.

## Prediction example

From the **inference repository root**, for `inputs/digit.png`:

```bash
docker compose run --rm backend /inputs/digit.png
```

The successful output has this shape; values depend on the image and model:

```json
{"prediction": 7, "confidence": 0.98, "model_version": "1"}
```

- `prediction`: predicted digit, from 0 to 9.
- `confidence`: softmax probability for that digit, not a calibrated accuracy score.
- `model_version`: exact numeric registry version loaded for this invocation.

Use one light digit on a dark background. PNG/JPEG input is converted, resized,
and normalized according to the registered model's metadata. There is no automatic inversion, cropping, or
centering. A model trained only for a smoke test may produce poor predictions.

## Further documentation

- [MLflow and shared-server connectivity](docs/mlflow.md)
- [Environment variables and advanced configuration](docs/configuration.md)
- [Architecture, model loading, and selection mismatch](docs/architecture.md)
- [Development, tests, coverage, and CI](docs/development.md)
- [Troubleshooting and legacy model access](docs/troubleshooting.md)
- [Training repository](https://github.com/ohadjerci/ml-engineering-lab)
