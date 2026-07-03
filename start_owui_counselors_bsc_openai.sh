#!/usr/bin/env bash
# Starts two isolated OpenWebUI-compatible counselor servers using a BSC/remote
# OpenAI-compatible backend by default.
#
# Defaults from the email:
#   Model endpoint: http://127.0.0.1:58091/v1
#   Model name:     Mistral-Small-3.1-24B-Instruct-2503
#   SDialog model:  openai:Mistral-Small-3.1-24B-Instruct-2503
#
# Run from the project root, for example:
#   cd /path/to/Sdialog
#   chmod +x ./start_owui_counselors_bsc_openai.sh
#   ./start_owui_counselors_bsc_openai.sh
#
# OpenWebUI endpoints:
#   http://localhost:1333/v1  -> university-counselor-ITA:latest
#   http://localhost:1334/v1  -> university-counselor-ENG:latest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODULE_NAME="${OWUI_SERVER_MODULE:-university_counseling.owui_session_server}"

HOST_ADDRESS="${OWUI_AGENT_HOST:-0.0.0.0}"
ITA_PORT="${OWUI_ITA_PORT:-1333}"
ENG_PORT="${OWUI_ENG_PORT:-1334}"

DEFAULT_MODEL="openai:Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_BASE="http://127.0.0.1:58091/v1"

ITA_MODEL="${COUNSELOR_MODEL_ITA:-$DEFAULT_MODEL}"
ENG_MODEL="${COUNSELOR_MODEL_ENG:-$DEFAULT_MODEL}"
ITA_BASE="${SDIALOG_OPENAI_API_BASE_ITA:-${OPENAI_API_BASE_ITA:-$DEFAULT_BASE}}"
ENG_BASE="${SDIALOG_OPENAI_API_BASE_ENG:-${OPENAI_API_BASE_ENG:-$DEFAULT_BASE}}"
API_KEY="${SDIALOG_OPENAI_API_KEY:-${OPENAI_API_KEY:-kk}}"

SESSION_TTL_SECONDS="${OWUI_SESSION_TTL_SECONDS:-3600}"
CLEANUP_INTERVAL_SECONDS="${OWUI_SESSION_CLEANUP_INTERVAL_SECONDS:-60}"
MAX_PATCH_HISTORY="${OWUI_MAX_PATCH_HISTORY:-80}"
MAX_PATCH_BUFFER="${OWUI_MAX_PATCH_BUFFER:-80}"
MAX_PATCH_EVENTS="${OWUI_MAX_PATCH_EVENTS:-80}"
MAX_DEBUG_SNAPSHOTS="${OWUI_MAX_DEBUG_SNAPSHOTS:-0}"
TOP_K="${SDIALOG_COUNSELOR_TOP_K:-6}"
CANDIDATE_POOL_SIZE="${SDIALOG_CANDIDATE_POOL_SIZE:-4}"
MAX_CTX_CHARS="${SDIALOG_MAX_CTX_CHARS:-2200}"
AGENT_THINK="${SDIALOG_AGENT_THINK:-0}"

HISTORY_TURNS="${SDIALOG_HISTORY_TURNS:-1}"

BSC_CONTEXT_BUDGET_GUARD="${BSC_CONTEXT_BUDGET_GUARD:-1}"
BSC_MAX_MODEL_LEN="${BSC_MAX_MODEL_LEN:-4096}"
BSC_OUTPUT_TOKEN_RESERVE="${BSC_OUTPUT_TOKEN_RESERVE:-512}"
BSC_MAX_INPUT_TOKENS="${BSC_MAX_INPUT_TOKENS:-3300}"
BSC_MAX_CHAT_MESSAGES="${BSC_MAX_CHAT_MESSAGES:-10}"
BSC_MAX_SYSTEM_CHARS="${BSC_MAX_SYSTEM_CHARS:-2500}"
BSC_MAX_MESSAGE_CHARS="${BSC_MAX_MESSAGE_CHARS:-1800}"
BSC_MAX_LAST_USER_CHARS="${BSC_MAX_LAST_USER_CHARS:-2400}"
BSC_MAX_LISTENER_TASK_CHARS="${BSC_MAX_LISTENER_TASK_CHARS:-3200}"
BSC_TRIM_LOG_ALWAYS="${BSC_TRIM_LOG_ALWAYS:-0}"

LOG_DIR="${OWUI_LOG_DIR:-$PROJECT_ROOT/logs}"
SKIP_CHECK=0
NO_WAIT=0
STOP_EXISTING=0

BSC_MAX_LISTENER_TASK_MESSAGES="${BSC_MAX_LISTENER_TASK_MESSAGES:-2}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --project-root PATH       Project root containing university_counseling/.
  --python PATH             Python executable. Default: python
  --host HOST               FastAPI bind host. Default: 0.0.0.0
  --ita-port PORT           ITA bridge port. Default: 1333
  --eng-port PORT           ENG bridge port. Default: 1334
  --ita-model MODEL         SDialog model for ITA. Default: $DEFAULT_MODEL
  --eng-model MODEL         SDialog model for ENG. Default: $DEFAULT_MODEL
  --ita-base URL            OpenAI-compatible /v1 endpoint for ITA. Default: $DEFAULT_BASE
  --eng-base URL            OpenAI-compatible /v1 endpoint for ENG. Default: $DEFAULT_BASE
  --api-key KEY             API key passed to the backend. Default: kk
  --top-k N                 RAG top_k. Default: $TOP_K
  --candidate-pool-size N   RAG candidate pool size. Default: $CANDIDATE_POOL_SIZE
  --max-ctx-chars N         Max RAG context chars. Default: $MAX_CTX_CHARS
  --think 0|1               Agent thinking. Default: 0
  --history-turns N         RAG/orchestrator history turns. Default: $HISTORY_TURNS
  --context-budget-guard 0|1 Enable context budget guard for vLLM/BSC. Default: $BSC_CONTEXT_BUDGET_GUARD
  --bsc-max-model-len N     Backend max_model_len. Default: $BSC_MAX_MODEL_LEN
  --bsc-output-token-reserve N Reserve output tokens. Default: $BSC_OUTPUT_TOKEN_RESERVE
  --bsc-max-input-tokens N  Max estimated input tokens. Default: $BSC_MAX_INPUT_TOKENS
  --bsc-max-chat-messages N Keep last N chat messages. Default: $BSC_MAX_CHAT_MESSAGES
  --bsc-max-system-chars N  Max chars for leading system prompt. Default: $BSC_MAX_SYSTEM_CHARS
  --bsc-max-message-chars N Max chars for normal messages. Default: $BSC_MAX_MESSAGE_CHARS
  --bsc-max-last-user-chars N Max chars for latest user msg. Default: $BSC_MAX_LAST_USER_CHARS
  --bsc-max-listener-task-chars N Max chars for listener task. Default: $BSC_MAX_LISTENER_TASK_CHARS
  --bsc-trim-log-always 0|1 Always log context trimming. Default: $BSC_TRIM_LOG_ALWAYS  
  --log-dir PATH            Log directory. Default: $LOG_DIR
  --skip-check              Do not test backend endpoints before start.
  --no-wait                 Start processes and return immediately.
  --stop-existing           Kill processes listening on the target ports before starting.
  -h, --help                Show this help.

Examples:
  $0
  $0 --ita-base http://127.0.0.1:58099/v1 --ita-model openai:gemma-4-31B-it \\
     --eng-base http://127.0.0.1:58091/v1 --eng-model openai:Mistral-Small-3.1-24B-Instruct-2503
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --host) HOST_ADDRESS="$2"; shift 2 ;;
    --ita-port) ITA_PORT="$2"; shift 2 ;;
    --eng-port) ENG_PORT="$2"; shift 2 ;;
    --ita-model) ITA_MODEL="$2"; shift 2 ;;
    --eng-model) ENG_MODEL="$2"; shift 2 ;;
    --ita-base) ITA_BASE="$2"; shift 2 ;;
    --eng-base) ENG_BASE="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --candidate-pool-size) CANDIDATE_POOL_SIZE="$2"; shift 2 ;;
    --max-ctx-chars) MAX_CTX_CHARS="$2"; shift 2 ;;
    --think) AGENT_THINK="$2"; shift 2 ;;
    --history-turns) HISTORY_TURNS="$2"; shift 2 ;;
    --context-budget-guard) BSC_CONTEXT_BUDGET_GUARD="$2"; shift 2 ;;
    --bsc-max-model-len) BSC_MAX_MODEL_LEN="$2"; shift 2 ;;
    --bsc-output-token-reserve) BSC_OUTPUT_TOKEN_RESERVE="$2"; shift 2 ;;
    --bsc-max-input-tokens) BSC_MAX_INPUT_TOKENS="$2"; shift 2 ;;
    --bsc-max-chat-messages) BSC_MAX_CHAT_MESSAGES="$2"; shift 2 ;;
    --bsc-max-system-chars) BSC_MAX_SYSTEM_CHARS="$2"; shift 2 ;;
    --bsc-max-message-chars) BSC_MAX_MESSAGE_CHARS="$2"; shift 2 ;;
    --bsc-max-last-user-chars) BSC_MAX_LAST_USER_CHARS="$2"; shift 2 ;;
    --bsc-max-listener-task-chars) BSC_MAX_LISTENER_TASK_CHARS="$2"; shift 2 ;;
    --bsc-trim-log-always) BSC_TRIM_LOG_ALWAYS="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --no-wait) NO_WAIT=1; shift ;;
    --stop-existing) STOP_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -d "$PROJECT_ROOT/university_counseling" ]]; then
  echo "Package folder not found: $PROJECT_ROOT/university_counseling" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

check_backend() {
  local label="$1"
  local base="$2"
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found; skipping $label backend check."
    return 0
  fi
  echo "$label backend check: $base"
  if curl -fsS --max-time 8 "$base/models" >/dev/null 2>&1; then
    echo "  OK: $base/models"
  else
    echo "  WARNING: could not GET $base/models. This may be fine if the server does not expose /models." >&2
    echo "  Try a manual chat completion test if startup later fails." >&2
  fi
}

stop_existing_on_port() {
  local port="$1"
  if [[ "$STOP_EXISTING" != "1" ]]; then
    return 0
  fi
  echo "Stopping existing processes on TCP port $port, if any."
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  elif command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN || true)"
    if [[ -n "$pids" ]]; then
      kill -9 $pids || true
    fi
  else
    echo "  WARNING: neither fuser nor lsof is available; cannot stop port $port automatically." >&2
  fi
}

PIDS=()
cleanup() {
  if [[ "$NO_WAIT" == "1" ]]; then
    return 0
  fi
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping pid=$pid"
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_one() {
  local lang="$1"
  local port="$2"
  local model="$3"
  local base="$4"
  local legacy_target="$5"
  local lower
  lower="$(echo "$lang" | tr '[:upper:]' '[:lower:]')"
  local stdout="$LOG_DIR/counselor_${lower}_${port}.out.log"
  local stderr="$LOG_DIR/counselor_${lower}_${port}.err.log"

  (
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
    export OWUI_PROCESS_MODEL="$lang"
    export OWUI_AGENT_PORT="$port"
    export OWUI_AGENT_HOST="$HOST_ADDRESS"
    export OWUI_LEGACY_MODEL_TARGET="$legacy_target"
    export COUNSELOR_MODEL="$model"
    export SDIALOG_MODEL_URI="$model"
    export SDIALOG_OPENAI_API_BASE="$base"
    export OPENAI_API_BASE="$base"
    export OPENAI_BASE_URL="$base"
    export SDIALOG_OPENAI_API_KEY="$API_KEY"
    export OPENAI_API_KEY="$API_KEY"
    export OWUI_SESSION_TTL_SECONDS="$SESSION_TTL_SECONDS"
    export OWUI_SESSION_CLEANUP_INTERVAL_SECONDS="$CLEANUP_INTERVAL_SECONDS"
    export OWUI_MAX_PATCH_HISTORY="$MAX_PATCH_HISTORY"
    export OWUI_MAX_PATCH_BUFFER="$MAX_PATCH_BUFFER"
    export OWUI_MAX_PATCH_EVENTS="$MAX_PATCH_EVENTS"
    export OWUI_MAX_DEBUG_SNAPSHOTS="$MAX_DEBUG_SNAPSHOTS"
    export SDIALOG_DEBUG=0
    export SDIALOG_AGENT_THINK="$AGENT_THINK"
    export SDIALOG_COUNSELOR_TOP_K="$TOP_K"
    export SDIALOG_CANDIDATE_POOL_SIZE="$CANDIDATE_POOL_SIZE"
    export SDIALOG_MAX_CTX_CHARS="$MAX_CTX_CHARS"
    export SDIALOG_HISTORY_TURNS="$HISTORY_TURNS"

    export BSC_CONTEXT_BUDGET_GUARD="$BSC_CONTEXT_BUDGET_GUARD"
    export BSC_MAX_MODEL_LEN="$BSC_MAX_MODEL_LEN"
    export BSC_OUTPUT_TOKEN_RESERVE="$BSC_OUTPUT_TOKEN_RESERVE"
    export BSC_MAX_INPUT_TOKENS="$BSC_MAX_INPUT_TOKENS"
    export BSC_MAX_CHAT_MESSAGES="$BSC_MAX_CHAT_MESSAGES"
    export BSC_MAX_SYSTEM_CHARS="$BSC_MAX_SYSTEM_CHARS"
    export BSC_MAX_MESSAGE_CHARS="$BSC_MAX_MESSAGE_CHARS"
    export BSC_MAX_LAST_USER_CHARS="$BSC_MAX_LAST_USER_CHARS"
    export BSC_MAX_LISTENER_TASK_CHARS="$BSC_MAX_LISTENER_TASK_CHARS"
    export BSC_TRIM_LOG_ALWAYS="$BSC_TRIM_LOG_ALWAYS"
    export BSC_MAX_LISTENER_TASK_MESSAGES="$BSC_MAX_LISTENER_TASK_MESSAGES"
    exec "$PYTHON_BIN" -m "$MODULE_NAME"
  ) >"$stdout" 2>"$stderr" &

  local pid=$!
  PIDS+=("$pid")
  echo "$lang server started: pid=$pid, port=$port, model=$model, base=$base"
  echo "  logs: $stdout | $stderr"
}

if [[ "$SKIP_CHECK" != "1" ]]; then
  check_backend "ITA" "$ITA_BASE"
  if [[ "$ENG_BASE" != "$ITA_BASE" ]]; then
    check_backend "ENG" "$ENG_BASE"
  fi
  echo ""
fi

stop_existing_on_port "$ITA_PORT"
stop_existing_on_port "$ENG_PORT"

start_one ITA "$ITA_PORT" "$ITA_MODEL" "$ITA_BASE" ITA
start_one ENG "$ENG_PORT" "$ENG_MODEL" "$ENG_BASE" ENG

sleep 2
for pid in "${PIDS[@]}"; do
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    echo "A counselor server exited immediately. Recent error logs:" >&2
    tail -n 80 "$LOG_DIR"/*.err.log 2>/dev/null || true
    exit 1
  fi
done

echo ""
echo "OpenWebUI endpoints:"
echo "  ITA: http://localhost:${ITA_PORT}/v1  -> university-counselor-ITA:latest"
echo "  ENG: http://localhost:${ENG_PORT}/v1  -> university-counselor-ENG:latest"
echo ""
echo "Health checks:"
echo "  curl http://127.0.0.1:${ITA_PORT}/health"
echo "  curl http://127.0.0.1:${ENG_PORT}/health"

if [[ "$NO_WAIT" == "1" ]]; then
  echo ""
  echo "NoWait enabled. Processes remain running in the background."
  trap - EXIT INT TERM
  exit 0
fi

echo ""
echo "Press Ctrl+C to stop both servers."
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "A counselor server exited unexpectedly. Check logs in $LOG_DIR" >&2
      tail -n 80 "$LOG_DIR"/*.err.log 2>/dev/null || true
      exit 1
    fi
  done
  sleep 2
done
