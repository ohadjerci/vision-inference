FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.12-slim AS base
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS runtime
ARG TORCH_VARIANT=cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 inference
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --extra ${TORCH_VARIANT} --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --extra ${TORCH_VARIANT} --no-dev --no-editable && uv pip check \
    && mkdir -p /home/inference/.cache/models \
    && chown -R inference:inference /home/inference
ENV PATH=/app/.venv/bin:$PATH
USER inference
ENTRYPOINT ["vision-infer"]

FROM runtime AS tests
ARG TORCH_VARIANT=cpu
USER root
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --extra ${TORCH_VARIANT} --no-editable
COPY tests ./tests
ENTRYPOINT []
CMD ["uv", "run", "--locked", "--extra", "cpu", "--no-editable", "pytest", "-ra"]
