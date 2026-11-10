# Development, tests, and CI

Run commands from the **inference repository root**. All backend checks run on CPU
and do not need the shared MLflow server, its network, or a real registered model.
The Docker image installs the `src/` package non-editably for its runtime.

## Backend checks

```bash
docker compose run --build --rm tests
docker compose run --rm tests uv run --locked --extra cpu ruff check .
docker compose run --rm tests uv run --locked --extra cpu ruff format --check .
docker compose run --rm tests uv run --locked --extra cpu mypy
docker compose run --rm tests uv lock --check
docker compose run --rm tests uv pip check
```

The default test command covers configuration, alias/explicit-version selection,
real TorchScript package loading, preprocessing, prediction, and CLI behavior.
It uses mocked registry interactions and temporary local artifacts; this does not
replace a live HTTP download/prediction check against a real training server.

Pytest collects branch coverage and emits `coverage.xml` and `coverage.json` in
`/app` inside the test container. Noneditable package paths map back to
`src/inference_backend/` for reports. With `--rm`, those local reports disappear
unless copied out or written through a deliberate writable mount.

## CI gates and four-badge reports

[Backend CI](../.github/workflows/ci.yml) runs on pushes to all branches, PRs
against `main`, and manual dispatch. It checks the lockfile, installs the package
with locked dependencies, verifies installed dependencies, and runs CPU tests.
Ruff lint, Ruff formatting, and MyPy are independent checks. Ruff also covers the
local reporting action; MyPy's backend scope is configured in `pyproject.toml`.

The reusable local action
[quality-report](../.github/actions/quality-report/action.yml) requires overall
line and branch percentages to meet the measured floors in
[coverage-baseline.json](../.github/coverage-baseline.json). Missing reports,
inconsistent XML/JSON, or a regression below either floor fail verification.
Revisit baselines only with reviewed test/scope changes.

The summary is exactly four adjacent badges: **Coverage**, **PR coverage**,
**Mypy**, and **Ruff**, without extra text. Overall coverage includes line and
branch percentages. PR coverage measures covered changed executable source lines;
it is informational, and is `N/A` for pushes or no executable-line changes. Tool
badges include MyPy error/file counts and Ruff lint/format counts.

The [comment workflow](../.github/workflows/ci-comment.yml) consumes a validated
numeric report after PR runs and creates or updates one badge-only bot comment.
It uses trusted default-branch code and never runs PR code with comment-writing
permissions. Both the workflow and action must first exist on the default branch.
Push runs alone do not post PR comments.

JUnit (`pytest-cpu`), XML/JSON (`coverage-cpu`), and numeric statistics
(`quality-report`) are uploaded for seven days. A failed dependency setup may
produce no coverage or JUnit report. Adding these workflows does not configure
GitHub branch protection or prove a hosted run has passed.

## Reporting-action checks

The backend Docker build context does not include root `.github/`. To test the
report action locally, mount that directory read-only:

```bash
docker compose run --rm -v "$PWD/.github/actions:/quality-actions:ro" tests \
  uv run --locked --extra cpu python -m unittest discover \
  -s /quality-actions/quality-report -p 'test_*.py'
```

To include action scripts in a local lint check matching CI:

```bash
docker compose run --rm -v "$PWD/.github/actions:/quality-actions:ro" tests \
  uv run --locked --extra cpu ruff check . /quality-actions/quality-report --config pyproject.toml
```

Use `ruff format --check` with the same paths/configuration for its formatting check.

## Dependency maintenance and rebuilds

The `lock` service mounts `backend/` and writes the lockfile as the configured
host user. After dependency changes:

```bash
LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" docker compose run --build --rm lock uv lock
docker compose run --build --rm tests
```

Commit `backend/uv.lock` together with `backend/pyproject.toml`. Images copy source
at build time; rebuild `backend` before checking CLI behavior after source edits:

```bash
docker compose build backend
```

The default image and GitHub CI install the `cpu` extra; the GPU override builds
the `cuda` extra. For local uv commands, select exactly one extra, for example
`uv sync --locked --extra cpu` or `uv sync --locked --extra cuda`, and pass the
same extra to `uv run`. The runtime stays non-root. Never add training checkout imports or
mounts to make a model load; publish a compatible standalone artifact instead.
