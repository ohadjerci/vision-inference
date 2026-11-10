set positional-arguments

default:
    @just --list

build device="cpu":
    case "$1" in cpu) docker compose build backend ;; cuda|gpu) docker compose -f compose.yaml -f compose.gpu.yaml build backend ;; *) echo "Device must be cpu or gpu" >&2; exit 1 ;; esac

run image device="cpu":
    case "$2" in cpu) docker compose run --rm backend --device cpu "$1" ;; cuda|gpu) docker compose -f compose.yaml -f compose.gpu.yaml run --build --rm backend --device cuda "$1" ;; *) echo "Device must be cpu or gpu" >&2; exit 1 ;; esac

debug-up device="cpu":
    case "$1" in cpu) docker compose up --build -d debug ;; cuda|gpu) docker compose -f compose.yaml -f compose.gpu.yaml up --build -d debug ;; *) echo "Device must be cpu or gpu" >&2; exit 1 ;; esac

debug-down:
    docker compose stop debug

bash:
    docker compose exec debug bash

debug image device="cpu":
    case "$2" in cpu) device=cpu ;; cuda|gpu) device=cuda ;; *) echo "Device must be cpu or gpu" >&2; exit 1 ;; esac; docker compose exec debug /app/.venv/bin/python -X frozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client /app/.venv/bin/vision-infer --device "$device" "$1"

api-up device="cpu":
    case "$1" in cpu) docker compose up --build -d api ;; cuda|gpu) docker compose -f compose.yaml -f compose.gpu.yaml up --build -d api ;; *) echo "Device must be cpu or gpu" >&2; exit 1 ;; esac

api-down:
    docker compose stop api
