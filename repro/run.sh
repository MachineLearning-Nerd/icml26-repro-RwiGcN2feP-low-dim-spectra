#!/usr/bin/env bash
# Fixed reproduction command (identical on every experiment node). CPU only.
# Reuses the single repository-level .venv; installs the uv tool if missing.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen --no-progress
uv run python -m repro.verify_all
