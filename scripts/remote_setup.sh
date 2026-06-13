#!/usr/bin/env bash
# First-time remote setup. Run once: `bash scripts/remote_setup.sh`
#
# On this remote, only `workspace/` persists across sessions. So we keep uv, the
# virtualenv, the package cache, the managed Python, and the HF dataset cache all
# INSIDE the repo directory (which you cloned under workspace/). One setup, then
# every later session just sources scripts/remote_session.sh.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UV_INSTALL_DIR="$REPO/.uv-bin"          # the uv binary itself
export UV_CACHE_DIR="$REPO/.uv-cache"          # downloaded wheels
export UV_PYTHON_INSTALL_DIR="$REPO/.uv-python" # uv-managed Python
export HF_HOME="$REPO/.hf"                      # MBPP / C4 dataset cache
export PATH="$UV_INSTALL_DIR:$PATH"
export INSTALLER_NO_MODIFY_PATH=1              # don't touch ~/.bashrc (won't persist anyway)

if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv into $UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo ">> creating venv (Python 3.11) at $REPO/.venv"
uv venv --python 3.11 "$REPO/.venv"
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

echo ">> installing CPU-only torch (no GPU on this box; avoids the CUDA build)"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ">> installing project (editable)"
uv pip install -e "$REPO"

echo
echo "Setup complete. Next:"
echo "  1) cp env.sh.example env.sh && nano env.sh   # add TINKER_API_KEY"
echo "  2) source scripts/remote_session.sh          # do this every new session"
