#!/usr/bin/env bash
set -euo pipefail

# Start SDialog/OpenWebUI session server with either:
#   - an OpenAI-compatible backend, e.g. openai:gemma-3-27b-it-q8_0
#   - an Ollama backend, e.g. ollama:qwen3:30b-thinking
#
# Examples:
#   ./start_sdialog_server.sh --provider openai --model gemma-3-27b-it-q8_0 --base-url http://127.0.0.1:10007/v1
#   ./start_sdialog_server.sh --provider ollama --model qwen3:30b-thinking --ollama-host http://127.0.0.1:11434
#   ./start_sdialog_server.sh --provider ollama --model qwen3:30b-thinking --server gender
#
# OpenWebUI should point to:
#   Base URL: http://<host>:1333/v1
#   API Key: anything/empty
#   Model ID: university-counselor:latest
# For --server gender, model IDs are usually:
#   university-counselor-male:latest
#   university-counselor-female:latest

PROJECT_DIR="$HOME//Sdialog"
PROVIDER="openai"
MODEL="gemma-3-27b-it-q8_0"
BASE_URL="http://127.0.0.1:10007/v1"
OLLAMA_HOST="http://127.0.0.1:11434"
SERVER="single"
MODULE=""
HOST="0.0.0.0"
PORT="1333"
PYTHON_EXE=""
DIALOG_LANGUAGE="Italian"
LANCEDB_DIR=""
LANCEDB_TABLE="universities"
OWUI_MODEL_ID=""
FORCED_GENDER=""
OPENAI_API_KEY_VALUE="kk"
REQUIRE_BACKEND="0"
SKIP_BACKEND_CHECK="0"

usage() {
  cat <<USAGE
Usage: $0 [options]

Provider options:
  --provider openai|ollama       Backend provider. Default: openai
  --model NAME                   Model name without provider prefix, or full URI.
                                 OpenAI default: gemma-3-27b-it-q8_0
                                 Ollama example: qwen3:30b-thinking
  --base-url URL                 OpenAI-compatible base URL, e.g. http://127.0.0.1:10007/v1
  --ollama-host URL              Ollama host without /v1, e.g. http://127.0.0.1:11434

Server options:
  --server single|gender         Python server to launch. Default: single
  --module MODULE                Override Python module explicitly
  --project-dir PATH             Project root. Default: $PROJECT_DIR
  --host HOST                    Server bind host. Default: 0.0.0.0
  --port PORT                    Server port. Default: 1333
  --owui-model-id ID             OpenWebUI-visible model ID/base ID
  --forced-gender male|female    Only for gender server: expose one forced-gender model

RAG options:
  --lancedb-dir PATH             LanceDB directory. Default: <project>/RAG_University_Eng/Embeddings/lancedb
  --lancedb-table NAME           LanceDB table. Default: universities
  --dialog-language LANG         Default: Italian

Checks:
  --require-backend              Exit if backend health check fails
  --skip-backend-check           Do not test backend before launch

Other:
  --python EXE                   Python executable. Default: venv Python if present, else python3.12
  -h, --help                     Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --base-url) BASE_URL="${2:-}"; shift 2 ;;
    --ollama-host) OLLAMA_HOST="${2:-}"; shift 2 ;;
    --server) SERVER="${2:-}"; shift 2 ;;
    --module) MODULE="${2:-}"; shift 2 ;;
    --project-dir) PROJECT_DIR="${2:-}"; shift 2 ;;
    --host) HOST="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --python) PYTHON_EXE="${2:-}"; shift 2 ;;
    --dialog-language) DIALOG_LANGUAGE="${2:-}"; shift 2 ;;
    --lancedb-dir) LANCEDB_DIR="${2:-}"; shift 2 ;;
    --lancedb-table) LANCEDB_TABLE="${2:-}"; shift 2 ;;
    --owui-model-id) OWUI_MODEL_ID="${2:-}"; shift 2 ;;
    --forced-gender) FORCED_GENDER="${2:-}"; shift 2 ;;
    --openai-api-key) OPENAI_API_KEY_VALUE="${2:-}"; shift 2 ;;
    --require-backend) REQUIRE_BACKEND="1"; shift ;;
    --skip-backend-check) SKIP_BACKEND_CHECK="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "$PROVIDER" in
  openai|ollama) ;;
  *) echo "ERROR: --provider must be openai or ollama" >&2; exit 2 ;;
esac

case "$SERVER" in
  single|gender) ;;
  *) echo "ERROR: --server must be single or gender" >&2; exit 2 ;;
esac

if [[ -z "$MODULE" ]]; then
  if [[ "$SERVER" == "gender" ]]; then
    MODULE="university_counseling.owui_session_server_two_gender_models"
  else
    MODULE="university_counseling.owui_session_server"
  fi
fi

if [[ -z "$LANCEDB_DIR" ]]; then
  LANCEDB_DIR="$PROJECT_DIR/RAG_University_Eng/Embeddings/lancedb"
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"

if [[ -z "$PYTHON_EXE" ]]; then
  if [[ -x "$PROJECT_DIR/venv/bin/python3.12" ]]; then
    PYTHON_EXE="$PROJECT_DIR/venv/bin/python3.12"
  elif [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON_EXE="$PROJECT_DIR/venv/bin/python"
  else
    PYTHON_EXE="python3.12"
  fi
fi

# Activate venv for PATH and prompt convenience, but still use PYTHON_EXE above.
if [[ -f "$PROJECT_DIR/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/venv/bin/activate"
fi

# Common SDialog/OpenWebUI server settings.
export OWUI_AGENT_HOST="$HOST"
export OWUI_AGENT_PORT="$PORT"
export DIALOG_LANGUAGE="$DIALOG_LANGUAGE"
export SDIALOG_ENABLE_RAG="1"
export SDIALOG_RAG_BACKEND="lancedb"
export SDIALOG_LANCEDB_DIR="$LANCEDB_DIR"
export SDIALOG_LANCEDB_TABLE="$LANCEDB_TABLE"
export OPENAI_API_KEY="$OPENAI_API_KEY_VALUE"
export SDIALOG_OPENAI_API_KEY="$OPENAI_API_KEY_VALUE"

if [[ -n "$OWUI_MODEL_ID" ]]; then
  export OWUI_MODEL_ID="$OWUI_MODEL_ID"
fi

if [[ -n "$FORCED_GENDER" ]]; then
  export OWUI_FORCED_GENDER="$FORCED_GENDER"
else
  unset OWUI_FORCED_GENDER || true
fi

# Build the model URI.
if [[ "$PROVIDER" == "openai" ]]; then
  if [[ "$MODEL" == openai:* || "$MODEL" == ollama:* ]]; then
    MODEL_URI="$MODEL"
  else
    MODEL_URI="openai:$MODEL"
  fi
  export SDIALOG_OPENAI_API_BASE="$BASE_URL"
  export OPENAI_API_BASE="$BASE_URL"
  export OPENAI_BASE_URL="$BASE_URL"
else
  if [[ "$MODEL" == openai:* || "$MODEL" == ollama:* ]]; then
    MODEL_URI="$MODEL"
  else
    MODEL_URI="ollama:$MODEL"
  fi
  export OLLAMA_HOST="$OLLAMA_HOST"
  export SDIALOG_OPENAI_API_BASE="${OLLAMA_HOST%/}/v1"
  export OPENAI_API_BASE="$SDIALOG_OPENAI_API_BASE"
  export OPENAI_BASE_URL="$SDIALOG_OPENAI_API_BASE"
fi

export SDIALOG_MODEL_URI="$MODEL_URI"
export COUNSELOR_MODEL="$MODEL_URI"

check_url=""
if [[ "$PROVIDER" == "openai" ]]; then
  check_url="${BASE_URL%/}/models"
else
  check_url="${OLLAMA_HOST%/}/api/tags"
fi

if [[ "$SKIP_BACKEND_CHECK" != "1" ]]; then
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 3 "$check_url" >/dev/null; then
      echo "Backend check OK: $check_url"
    else
      echo "WARNING: backend check failed: $check_url" >&2
      echo "         The SDialog server can start, but chat completions will fail until the LLM backend is reachable." >&2
      if [[ "$REQUIRE_BACKEND" == "1" ]]; then
        exit 1
      fi
    fi
  else
    echo "WARNING: curl not found; skipping backend check." >&2
  fi
fi

if [[ ! -d "$SDIALOG_LANCEDB_DIR" ]]; then
  echo "WARNING: LanceDB directory not found: $SDIALOG_LANCEDB_DIR" >&2
elif [[ ! -e "$SDIALOG_LANCEDB_DIR/$SDIALOG_LANCEDB_TABLE.lance" ]]; then
  echo "WARNING: LanceDB table not found: $SDIALOG_LANCEDB_DIR/$SDIALOG_LANCEDB_TABLE.lance" >&2
fi

cat <<INFO

Starting SDialog OpenWebUI session server
  Project:       $PROJECT_DIR
  Python:        $PYTHON_EXE
  Module:        $MODULE
  Bind:          http://$HOST:$PORT
  Provider:      $PROVIDER
  Model URI:     $MODEL_URI
  LLM base URL:  $OPENAI_BASE_URL
  LanceDB:       $SDIALOG_LANCEDB_DIR/$SDIALOG_LANCEDB_TABLE.lance

OpenWebUI connection:
  Base URL:      http://<this-host>:$PORT/v1
  API Key:       empty or any value

INFO

exec "$PYTHON_EXE" -m "$MODULE"
