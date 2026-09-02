#!/usr/bin/env bash
set -euo pipefail

CTM_VENV_PATH="${CTM_VENV_PATH:-/home/madao/.venvs/ctm-dreamer}"
if [[ ! -x "${CTM_VENV_PATH}/bin/python" ]]; then
  echo "Missing WSL virtual environment: ${CTM_VENV_PATH}" >&2
  echo "Create it and install requirements.txt first." >&2
  exit 1
fi

CTM_SITE_PACKAGES="${CTM_VENV_PATH}/lib/python3.12/site-packages"
if [[ -d "${CTM_SITE_PACKAGES}/nvidia" ]]; then
  CTM_CUDA_LIBS="$(find "${CTM_SITE_PACKAGES}/nvidia" -type d -name lib -print | paste -sd: -)"
  export LD_LIBRARY_PATH="${CTM_CUDA_LIBS}:${LD_LIBRARY_PATH:-}"
fi

export MUJOCO_GL="${MUJOCO_GL:-glfw}"
exec "${CTM_VENV_PATH}/bin/python" "$@"
