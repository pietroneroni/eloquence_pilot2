#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${PY_SCRIPT:-$SCRIPT_DIR/owui_session_server_split.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Shared defaults. Override from shell or .env before launching.
export OWUI_AGENT_HOST="${OWUI_AGENT_HOST:-0.0.0.0}"
export OWUI_SESSION_TTL_SECONDS="${OWUI_SESSION_TTL_SECONDS:-3600}"
export OWUI_SESSION_CLEANUP_INTERVAL_SECONDS="${OWUI_SESSION_CLEANUP_INTERVAL_SECONDS:-60}"
export OWUI_MAX_PATCH_HISTORY="${OWUI_MAX_PATCH_HISTORY:-80}"
export OWUI_MAX_PATCH_BUFFER="${OWUI_MAX_PATCH_BUFFER:-80}"
export OWUI_MAX_PATCH_EVENTS="${OWUI_MAX_PATCH_EVENTS:-80}"
export OWUI_MAX_DEBUG_SNAPSHOTS="${OWUI_MAX_DEBUG_SNAPSHOTS:-0}"
export SDIALOG_DEBUG="${SDIALOG_DEBUG:-0}"

# RAG/latency defaults. Tune these after measuring.
export SDIALOG_COUNSELOR_TOP_K="${SDIALOG_COUNSELOR_TOP_K:-6}"
export SDIALOG_CANDIDATE_POOL_SIZE="${SDIALOG_CANDIDATE_POOL_SIZE:-4}"
export SDIALOG_MAX_CTX_CHARS="${SDIALOG_MAX_CTX_CHARS:-2200}"

# If both processes use the same Ollama/OpenAI-compatible endpoint, set once here.
# Example: export SDIALOG_OPENAI_API_BASE=http://127.0.0.1:11434/v1
export SDIALOG_OPENAI_API_BASE="${SDIALOG_OPENAI_API_BASE:-${OPENAI_API_BASE:-http://127.0.0.1:10007/v1}}"

# Model backends used by SDialog. Override if ITA and ENG use different Ollama models.
export COUNSELOR_MODEL_ITA="${COUNSELOR_MODEL_ITA:-${COUNSELOR_MODEL:-${SDIALOG_MODEL_URI:-openai:gemma-3-27b-it-q8_0}}}"
export COUNSELOR_MODEL_ENG="${COUNSELOR_MODEL_ENG:-${COUNSELOR_MODEL:-${SDIALOG_MODEL_URI:-openai:gemma-3-27b-it-q8_0}}}"

start_ita() {
  (
    export OWUI_PROCESS_MODEL=ITA
    export OWUI_AGENT_PORT="${OWUI_ITA_PORT:-1333}"
    export OWUI_LEGACY_MODEL_TARGET=ITA
    exec "$PYTHON_BIN" "$PY_SCRIPT"
  ) &
  PID_ITA=$!
  echo "ITA server pid=$PID_ITA port=${OWUI_ITA_PORT:-1333}"
}

start_eng() {
  (
    export OWUI_PROCESS_MODEL=ENG
    export OWUI_AGENT_PORT="${OWUI_ENG_PORT:-1334}"
    export OWUI_LEGACY_MODEL_TARGET=ENG
    exec "$PYTHON_BIN" "$PY_SCRIPT"
  ) &
  PID_ENG=$!
  echo "ENG server pid=$PID_ENG port=${OWUI_ENG_PORT:-1334}"
}

stop_children() {
  echo "Stopping servers..."
  jobs -p | xargs -r kill
}

trap stop_children INT TERM EXIT
start_ita
start_eng
wait
