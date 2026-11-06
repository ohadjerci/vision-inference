# Shared MLflow connectivity

## Start with the training project

The [training repository](https://github.com/ohadjerci/ml-engineering-lab) owns the
MLflow service, database, artifact storage, and Docker network. Follow its README
quick start to create writable storage, start MLflow, and train/register a model.
Its `docs/mlflow.md` is the detailed server/storage guide; its `docs/migration.md`
explains historical file-store handling. Do not create a second server here.

For subsequent starts, run from the **training repository root**:

```bash
docker compose up --build -d mlflow
docker compose ps mlflow
```

The server must be healthy and have a compatible `MNIST` registry version.
Check `best_model` on that registered model in the UI, or select an explicit
numeric version as described in [configuration](configuration.md#model-selection).

## Host and container addresses

| Where the client runs | URL |
| --- | --- |
| Browser or host client | `http://localhost:50001` |
| Inference container on `vision-mlflow` | `http://mlflow:5000` |

Inside the inference container, `localhost` refers to the inference container,
not the training server. The training Compose project creates the network;
this project's `backend` service joins it as an external network.

From the **inference repository root**, check that the network exists:

```bash
docker network inspect vision-mlflow
```

This same-host setup is the implemented default. A server on another computer
needs a reachable HTTP(S) address and corresponding server/network configuration;
changing a URL does not create that connectivity automatically.

## Artifact downloads and lifecycle

The training server stores registry/tracking metadata in SQLite and serves model
artifacts through HTTP. Inference resolves an alias or numeric version via the
registry, then downloads the exact version's packaged artifacts. No training
source, database, or log directory is mounted into the inference container.

`inputs/` is mounted read-only at `/inputs`. Downloads use the `model-cache`
named volume at `/home/inference/.cache/models`. Keep MLflow running while clients
need to resolve or download models; a cache directory is not an offline fallback.
Stopping or removing an inference container does not stop the shared server.

The current experiment name on the training side is `project_http`. Experiments
organize runs; inference looks up the `MNIST` registered model, not an experiment
name. Older experiment names can remain without affecting a numeric registry
lookup, provided the selected version's artifact is remotely accessible.
