#!/usr/bin/env bash
# Fixed reproduction command (identical on every experiment node).
# Reuses the single repository-level .venv; CPU only.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen --no-progress
uv run python -m repro.verify_all
