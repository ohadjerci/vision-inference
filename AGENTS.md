# Repository Guidelines

## Project Structure & Module Organization

This repository provides CPU inference for an MNIST model from a separate training repository. The interface is the `vision-infer` CLI; no web API exists.

- `backend/src/inference_backend/`: environment settings, MLflow model loading, image preprocessing, prediction, and CLI.
- `backend/tests/`: pytest suites and shared fixtures in `conftest.py`.
- `backend/pyproject.toml`, `backend/uv.lock`, and `backend/Dockerfile`: dependencies, tooling, and container builds.
- `compose.yaml`: runtime, test, and lockfile services.
- `inputs/`: local images mounted read-only; model downloads use a named Docker volume.
- `frontend/`: placeholder for future development.

## Build, Test, and Development Commands

Run these commands from the repository root:

```bash
cp .env.example .env
docker compose build backend
docker compose run --rm backend /inputs/digit.png
docker compose run --build --rm tests
docker compose run --rm tests uv run --locked --extra cpu ruff check .
docker compose run --rm tests uv run --locked --extra cpu ruff format --check .
docker compose run --rm tests uv run --locked --extra cpu mypy
docker compose run --rm tests uv lock --check
docker compose run --rm tests uv pip check
```

These initialize configuration, build/run inference, execute tests, and check lint, formatting, types, and dependencies. Inference requires the training repository's shared MLflow server and `vision-mlflow` network; checks run independently. Commit dependency changes with `backend/uv.lock`.

## Coding Style & Naming Conventions

Use Python 3.12, four-space indentation, and Ruff's 99-character line limit. Use `snake_case` for functions/modules and `PascalCase` for classes. Add type annotations to production interfaces. Keep loading and prediction reusable independently of the CLI. Run Ruff and MyPy before submitting changes.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Reuse fixtures, temporary paths, and mocked registry interactions. Cover preprocessing, model validation, and CLI success/error behavior when relevant. CI enforces the independently measured CPU line and branch floors in `.github/coverage-baseline.json`; do not lower them just to hide regressions. Live integration verification requires a published model and the separate training server.

## Commit & Pull Request Guidelines

“Commit & push” authorizes creating or updating a pull request automatically when safe, after local checks and successful remote CI for the exact pushed commit. Reuse an open PR for the current branch; otherwise create one against the repository's default branch. When starting on the default branch, create a feature branch before committing unless the user explicitly requests a direct push. Do not merge automatically, rewrite published history, create duplicate/empty PRs, or manufacture a retrospective PR for changes already on the target branch. Resolve the head/base from repository context first; if a safe action remains blocked or ambiguous, explain the specific reason and ask for direction before taking a risky action. Report the PR URL, branch, commit SHA, validation results, and exact CI run URL. PR creation is part of delivery, not a workaround solely to trigger CI.

Existing commits use short imperative subjects, such as “Add standalone MLflow inference CLI.” Follow that style. Describe behavior changes and validation in pull requests; link related issues when applicable. CI runs on all branch pushes, PRs targeting `main`, and manual dispatch. It checks tests, Ruff, configured MyPy, locked dependencies, and measured line/branch coverage floors in `.github/coverage-baseline.json`. Pytest emits branch-enabled XML/JSON coverage and JUnit reports; artifacts are retained for seven days. See `docs/development.md` for commands and artifact names.

## Configuration & Agent Instructions

Never commit `.env`, inputs, model downloads, or generated outputs. Load artifacts only from a trusted registry.

When authoritative third-party documentation is needed, check installed versions and consult Context7 automatically. Prefer compatible documentation; use another reliable source if unavailable. Skip documentation lookups for simple changes and never invent API signatures or configuration parameters.

## CI Reporting

Use the reusable `.github/actions/quality-report` action. The Actions summary and PR comment must be exactly four adjacent badges labeled `Coverage`, `PR coverage`, `Mypy`, and `Ruff`, without extra text. Coverage includes overall line/branch percentages; PR coverage is changed executable-line coverage (`N/A` when not applicable). Report MyPy error/file and Ruff lint/format counts.

The separate `ci-comment.yml` workflow runs trusted default-branch code, validates a numeric artifact from read-only PR CI, and creates or updates one bot comment. Never execute untrusted PR code with the write-scoped comment token. Commenting requires the trusted workflow/action on the default branch. Document actual checks run; adding a workflow does not establish a successful hosted CI result.
