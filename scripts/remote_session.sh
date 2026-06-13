#!/usr/bin/env bash
# Source at the START of every remote session: `source scripts/remote_session.sh`
# (Shell state doesn't persist across sessions, so this re-exports the paths that
# point at the persisted, in-repo uv/venv/caches and activates the environment.)

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_INSTALL_DIR="$_REPO/.uv-bin"
export UV_CACHE_DIR="$_REPO/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$_REPO/.uv-python"
export HF_HOME="$_REPO/.hf"
export PATH="$UV_INSTALL_DIR:$PATH"

if [ -f "$_REPO/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$_REPO/.venv/bin/activate"
else
  echo "No venv found — run: bash scripts/remote_setup.sh" >&2
fi

if [ -f "$_REPO/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$_REPO/env.sh"
else
  echo "No env.sh — run: cp env.sh.example env.sh && nano env.sh (add TINKER_API_KEY)" >&2
fi

echo "Ready: $(python --version 2>&1) | uv $(uv --version 2>/dev/null | awk '{print $2}') | HF_HOME=$HF_HOME"
