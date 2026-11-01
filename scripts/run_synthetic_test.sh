#!/usr/bin/env bash
# Run sequentially from any working directory. All CLI arguments are forwarded.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir/.."

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install uv, then run this script again." >&2
    exit 1
fi

exec uv run --locked --extra dev python -u -m src.synthetic_benchmark "$@"
