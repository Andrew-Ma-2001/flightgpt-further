#!/usr/bin/env bash
set -euo pipefail

echo "===== System ====="
python --version || true
uname -a || true
ldd --version || true
free -h || true
ulimit -a || true
echo "MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX-<unset>}"
nvidia-smi || true

echo
echo "===== Python Packages ====="
pip freeze | sort || true
conda list || true

echo
echo "===== Key Python Lib Versions ====="
python - <<'PY'
import importlib
pkgs = [
    "openai", "httpx", "requests", "aiohttp", "pydantic",
    "PIL", "numpy", "torch", "transformers", "datasets", "vllm",
]
for name in pkgs:
    try:
        m = importlib.import_module(name)
        print(f"{name}=={getattr(m, '__version__', '<unknown>')}")
    except Exception as e:
        print(f"{name}=<unavailable> ({e})")
PY
