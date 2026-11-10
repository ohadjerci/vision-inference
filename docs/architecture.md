# Architecture and model loading

## Backend responsibilities

The `backend/` directory owns the installed Python package, dependencies, tests,
and Docker build. It uses a conventional `src/inference_backend/` package layout;
there are no manual `sys.path` changes or imports from the training checkout.

| Module | Responsibility |
| --- | --- |
| `settings.py` | Validate environment settings and writable cache configuration. |
| `model_loader.py` | Resolve a model version, download its metadata and load on the selected device. |
| `predictor.py` | Prepare images, run inference, and return a prediction result. |
| `cli.py` | Parse an image path, call the backend functions, and print JSON or an error. |

Loading and prediction are independent of the CLI. The HTTP API loads one model at startup
and reuses it for requests from the React drawing interface.

## Prediction flow

1. Read settings and resolve `best_model`, unless a numeric version was supplied.
2. Pin that exact version and download its MLflow package through HTTP.
3. Load the network in evaluation mode on the selected CPU or CUDA device.
4. Validate metadata for image size, normalization and labels; validate input shape
   when present. Infer channels from per-channel normalization values when the
   shape is absent, using one channel for scalar normalization.
   Decode PNG/JPEG input, convert to grayscale or RGB, resize with bilinear
   interpolation, scale pixels to 0–1, and normalize using metadata values.
5. Run in inference mode and require the metadata's number of finite logits. Softmax gives class
   probabilities; return the highest-probability digit, confidence, and model version.

Successful CLI output is JSON on stdout. Errors go to stderr with a nonzero exit
status. Library callers can use `Settings`, `load_model`, and `predict` directly.

## Training-side registration

The training project tests its lowest-validation-loss checkpoint, packages a CPU
TorchScript network with preprocessing metadata, and registers a new `MNIST`
version. Standalone evaluation can publish the explicitly evaluated checkpoint.
The exported network is self-contained; Lightning and training modules are not
needed in this runtime.

## Selection requirement mismatch

**Agreed requirement:** promote `best_model` using validation results.

**Current training implementation:** validation loss chooses the checkpoint,
but registry promotion compares **`test/loss`**. A strictly smaller finite test
loss replaces the alias; ties keep the incumbent. Publishers must run sequentially.

This does not implement validation-based alias promotion. Inference neither ranks
runs nor corrects that policy; it follows the alias or explicit version. A separate
training-side code change is needed to meet the agreed requirement. The detailed
source locations are documented in the training project's `docs/architecture.md`.
