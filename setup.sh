#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Brev Launchable setup — NVIDIA Agent Toolkit workshop
#
# Runs once at first boot on a fresh CPU VM. Installs everything the notebook
# needs and registers a Jupyter kernel carrying the attendee's API key, so the
# notebook runs with no manual steps.
#
# Reads the launch parameter NVIDIA_API_KEY from the environment.
# Writes ~/.workshop_ready when, and only when, everything verified.
# ---------------------------------------------------------------------------
set -uo pipefail

# Set this before publishing the Launchable. Used only as a fallback, if Brev has
# not already cloned the repo by the time this script runs.
WORKSHOP_REPO_URL="${WORKSHOP_REPO_URL:-https://github.com/saurabh-nvidia/agent-toolkit-workshop}"

USER_HOME="${HOME:-/home/ubuntu}"
LOG="$USER_HOME/workshop_setup.log"
VENV="$USER_HOME/relayenv"
READY="$USER_HOME/.workshop_ready"
STATUS="$USER_HOME/.workshop_status"

exec > >(tee -a "$LOG") 2>&1
rm -f "$READY"
step() { echo "$1" > "$STATUS"; echo "=== $1 ==="; }

step "starting"
echo "$(date -u)"

# --- 1. uv (the base image ships Python 3.10; we need 3.12) -----------------
step "installing uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$USER_HOME/.local/bin:$PATH"
uv --version || { echo "FATAL: uv install failed"; exit 1; }

# --- 2. Python environment --------------------------------------------------
step "creating python 3.12 environment"
uv venv --python 3.12 "$VENV" || { echo "FATAL: venv failed"; exit 1; }

step "installing notebook dependencies"
uv pip install --python "$VENV/bin/python" \
  "nemo-relay[langgraph]" \
  langchain-openai \
  jupyterlab \
  ipykernel \
  pandas || { echo "FATAL: pip install failed"; exit 1; }

# --- 3. Switchyard ----------------------------------------------------------
# The published nemo-switchyard wheel omits two runtime dependencies (pyyaml,
# uvicorn), so the documented install command fails. Inject them explicitly.
# The docs also say --python 3.10; the package actually requires >= 3.12.
step "installing switchyard"
uv tool install --force \
  --with pyyaml --with uvicorn --with fastapi \
  --python 3.12 'nemo-switchyard[cli]' || echo "WARNING: switchyard install failed (section 6 will not run)"

# --- 4. Locate the workshop repo and expose the notebook --------------------
step "locating workshop content"
REPO=""
for candidate in "$USER_HOME"/*/notebook/agent_workshop.ipynb; do
  [ -f "$candidate" ] && REPO="$(dirname "$(dirname "$candidate")")" && break
done

# If Brev has not cloned the repo yet (ordering is not guaranteed), clone it here.
if [ -z "$REPO" ] && [ -n "${WORKSHOP_REPO_URL:-}" ]; then
  echo "repo not present - cloning $WORKSHOP_REPO_URL"
  git clone --depth 1 "$WORKSHOP_REPO_URL" "$USER_HOME/agent-toolkit-workshop" \
    && REPO="$USER_HOME/agent-toolkit-workshop"
fi

if [ -n "$REPO" ]; then
  echo "repo found at $REPO"
  ln -sf "$REPO/notebook/agent_workshop.ipynb" "$USER_HOME/agent_workshop.ipynb"
  ln -sf "$REPO/ATTENDEE_PREWORK.md"            "$USER_HOME/ATTENDEE_PREWORK.md" 2>/dev/null
  ln -sf "$REPO/benchmark"                       "$USER_HOME/benchmark" 2>/dev/null
  ln -sf "$REPO/optional"                        "$USER_HOME/optional"  2>/dev/null
else
  echo "WARNING: workshop notebook not found and no WORKSHOP_REPO_URL set."
  echo "         Set WORKSHOP_REPO_URL at the top of this script before publishing."
fi

# --- 5. Jupyter kernel ------------------------------------------------------
# Brev's Jupyter runs the host Python 3.10 and cannot see our venv. Register the
# venv as its own kernel and bake the API key into the kernel environment, so
# the notebook reads it from os.environ without the attendee doing anything.
step "registering jupyter kernel"
"$VENV/bin/python" -m ipykernel install --user \
  --name relayenv --display-name "Agent Workshop (Python 3.12)"

KERNEL_JSON="$USER_HOME/.local/share/jupyter/kernels/relayenv/kernel.json"
if [ -n "${NVIDIA_API_KEY:-}" ] && [ -f "$KERNEL_JSON" ]; then
  "$VENV/bin/python" - "$KERNEL_JSON" "$NVIDIA_API_KEY" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
spec = json.load(open(path))
spec.setdefault("env", {})["NVIDIA_API_KEY"] = key
json.dump(spec, open(path, "w"), indent=1)
print("API key injected into kernel environment")
PY
  printf "export NVIDIA_API_KEY='%s'\n" "$NVIDIA_API_KEY" > "$USER_HOME/.env_keys"
  chmod 600 "$USER_HOME/.env_keys"
else
  echo "WARNING: NVIDIA_API_KEY launch parameter was not supplied."
  echo "         The notebook will stop at the setup cell with instructions."
fi

# --- 6. Verify --------------------------------------------------------------
step "verifying"
"$VENV/bin/python" - <<'PY'
import importlib.util, sys
missing = [m for m in ("nemo_relay", "langgraph", "langchain_openai", "jupyterlab", "pandas")
           if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:", missing); sys.exit(1)
print("python dependencies OK")
PY
DEPS=$?

"$USER_HOME/.local/bin/switchyard" --version >/dev/null 2>&1 \
  && echo "switchyard OK" || echo "switchyard MISSING"

if [ "$DEPS" -eq 0 ]; then
  step "ready"
  touch "$READY"
  echo "=== SETUP COMPLETE $(date -u) ==="
else
  step "failed"
  echo "=== SETUP FAILED — see $LOG ==="
  exit 1
fi
