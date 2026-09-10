#!/usr/bin/env bash
set -euo pipefail

CTM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CTM_VENV_PATH="${CTM_VENV_PATH:-/home/madao/.venvs/ctm-dreamer}"
CTM_PYTHON="${CTM_VENV_PATH}/bin/python"
if [[ ! -x "${CTM_PYTHON}" ]]; then
  echo "Missing WSL Python: ${CTM_PYTHON}" >&2
  exit 1
fi

# Discover pip-installed CUDA libraries using this interpreter, not a fixed
# Python minor version. The WSL driver libraries are supplied by Windows.
CTM_CUDA_LIBS="$("${CTM_PYTHON}" -c 'import pathlib, site; print(":".join(sorted({str(p) for root in site.getsitepackages() for p in pathlib.Path(root).glob("nvidia/*/lib") if p.is_dir()})))')"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${CTM_CUDA_LIBS:+:${CTM_CUDA_LIBS}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--check-gpu" ]]; then
  shift
  exec "${CTM_PYTHON}" "${CTM_SCRIPT_DIR}/gpu_check.py" "$@"
fi
if [[ $# -eq 0 ]]; then
  echo "Usage: bash run_wsl.sh --check-gpu [--require-gpu]" >&2
  echo "       bash run_wsl.sh -u dreamer.py [training options]" >&2
  exit 2
fi
if [[ "${CTM_REQUIRE_GPU:-0}" == "1" ]]; then
  "${CTM_PYTHON}" "${CTM_SCRIPT_DIR}/gpu_check.py" --require-gpu
else
  "${CTM_PYTHON}" "${CTM_SCRIPT_DIR}/gpu_check.py"
fi
exec "${CTM_PYTHON}" "$@"
