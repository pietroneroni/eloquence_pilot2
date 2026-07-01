# university_counseling/owui_session_server.py
"""
Open WebUI-compatible session server for the university counselor.

This server exposes OpenWebUI chat models in one process,
or exactly one model per process. For production/performance, prefer one
process per exposed model because AgentWithRag still uses module-level state.

    university-counselor-ITA:latest  -> Italian dialogue + Italian RAG
    university-counselor-ENG:latest  -> English dialogue + English RAG

Run with OWUI_PROCESS_MODEL=ITA or OWUI_PROCESS_MODEL=ENG to isolate the
Italian and English agents in separate OS processes. Each process has its own
SDialog Agent instances, retriever cache, session map, and AgentWithRag globals.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sdialog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from sdialog.agents import Agent
from sdialog.personas import Persona

from university_counseling.agents_setup import (
    TimedRetriever,
    _build_expert_response_details,
    _configure_rag_env,
    sanitize_expert_output,
)
from university_counseling.social_practice import get_social_practice
import university_counseling.AgentWithRag as rag_state
from university_counseling.AgentWithRag import UniversityCounselorFlowOrchestrator


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PORT = int(os.getenv("OWUI_AGENT_PORT", "1333"))
HOST = os.getenv("OWUI_AGENT_HOST", "0.0.0.0")
SOCIAL_PRACTICE_NAME = os.getenv("SOCIAL_PRACTICE_NAME", "university_counseling")

# Which model(s) this process exposes. For production use two processes:
#
#   OWUI_PROCESS_MODEL=ITA OWUI_AGENT_PORT=1333 python owui_session_server_ollama_direct.py
#   OWUI_PROCESS_MODEL=ENG OWUI_AGENT_PORT=1334 python owui_session_server_ollama_direct.py
#
# Accepted values: ITA, ENG, BOTH. BOTH keeps backward compatibility, but it
# still serializes agent turns through RAG_GLOBAL_LOCK because AgentWithRag uses
# module-level state.
OWUI_PROCESS_MODEL = os.getenv("OWUI_PROCESS_MODEL", "BOTH").strip().upper()

# BSC/remote OpenAI-compatible backend defaults from the email.
# The /v1 endpoint is the model provider endpoint. OpenWebUI still connects to
# this FastAPI bridge on /v1, for example http://localhost:1333/v1.
DEFAULT_OPENAI_API_BASE = os.getenv(
    "SDIALOG_OPENAI_API_BASE",
    os.getenv("OPENAI_API_BASE", "http://127.0.0.1:58091/v1"),
).strip()
OPENAI_API_BASE_ITA = os.getenv(
    "SDIALOG_OPENAI_API_BASE_ITA",
    os.getenv("OPENAI_API_BASE_ITA", DEFAULT_OPENAI_API_BASE),
).strip()
OPENAI_API_BASE_ENG = os.getenv(
    "SDIALOG_OPENAI_API_BASE_ENG",
    os.getenv("OPENAI_API_BASE_ENG", DEFAULT_OPENAI_API_BASE),
).strip()
OPENAI_API_KEY = os.getenv("SDIALOG_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "kk"))

DEFAULT_COUNSELOR_MODEL = os.getenv(
    "COUNSELOR_MODEL",
    os.getenv("SDIALOG_MODEL_URI", "openai:Mistral-Small-3.1-24B-Instruct-2503"),
)
COUNSELOR_MODEL_ITA = os.getenv(
    "COUNSELOR_MODEL_ITA",
    os.getenv("SDIALOG_MODEL_URI_ITA", DEFAULT_COUNSELOR_MODEL),
)
COUNSELOR_MODEL_ENG = os.getenv(
    "COUNSELOR_MODEL_ENG",
    os.getenv("SDIALOG_MODEL_URI_ENG", DEFAULT_COUNSELOR_MODEL),
)


def _selected_backend_base() -> str:
    if OWUI_PROCESS_MODEL in {"ITA", "IT"}:
        return OPENAI_API_BASE_ITA
    if OWUI_PROCESS_MODEL in {"ENG", "EN"}:
        return OPENAI_API_BASE_ENG
    return DEFAULT_OPENAI_API_BASE


def _selected_counselor_model() -> str:
    if OWUI_PROCESS_MODEL in {"ITA", "IT"}:
        return COUNSELOR_MODEL_ITA
    if OWUI_PROCESS_MODEL in {"ENG", "EN"}:
        return COUNSELOR_MODEL_ENG
    return DEFAULT_COUNSELOR_MODEL


SELECTED_OPENAI_API_BASE = _selected_backend_base()
if SELECTED_OPENAI_API_BASE:
    os.environ["SDIALOG_OPENAI_API_BASE"] = SELECTED_OPENAI_API_BASE
    os.environ["OPENAI_API_BASE"] = SELECTED_OPENAI_API_BASE
    os.environ["OPENAI_BASE_URL"] = SELECTED_OPENAI_API_BASE
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

# Keep thinking off by default because many OpenAI-compatible servers do not
# support vendor-specific thinking parameters. Enable only if the backend model
# and server explicitly support it.
ENABLE_AGENT_THINKING = os.getenv("SDIALOG_AGENT_THINK", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# Keep a default SDialog model configured for code paths that rely on the global
# default. Each Agent below still receives its explicit per-session model.
sdialog.config.llm(_selected_counselor_model())

SESSION_TTL_SECONDS = int(os.getenv("OWUI_SESSION_TTL_SECONDS", str(60 * 60)))
SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv("OWUI_SESSION_CLEANUP_INTERVAL_SECONDS", "60"))
MAX_PATCH_HISTORY = int(os.getenv("OWUI_MAX_PATCH_HISTORY", "80"))
MAX_PATCH_BUFFER = int(os.getenv("OWUI_MAX_PATCH_BUFFER", "80"))
MAX_PATCH_EVENTS = int(os.getenv("OWUI_MAX_PATCH_EVENTS", "80"))
MAX_DEBUG_SNAPSHOTS = int(os.getenv("OWUI_MAX_DEBUG_SNAPSHOTS", "0"))
MAX_ADVICE_OPTIONS = int(os.getenv("OWUI_MAX_ADVICE_OPTIONS", "30"))

# Required while AgentWithRag stores per-conversation state as module globals. In
# one-model-per-process mode this lock only serializes sessions inside that one
# model process, not ITA vs ENG globally.
RAG_GLOBAL_LOCK = threading.RLock()

logging.basicConfig(
    level=os.getenv("OWUI_SESSION_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("owui-session-server")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    dialog_language: str
    rag_lang: str
    counselor_model: str


def _ensure_latest_suffix(model_id: str) -> str:
    model_id = (model_id or "").strip()
    return model_id if model_id.endswith(":latest") else f"{model_id}:latest"


ITA_MODEL_ID = _ensure_latest_suffix(os.getenv("OWUI_ITA_MODEL_ID", "university-counselor-ITA:latest"))
ENG_MODEL_ID = _ensure_latest_suffix(os.getenv("OWUI_ENG_MODEL_ID", "university-counselor-ENG:latest"))

ALL_MODEL_CONFIGS: Dict[str, ModelConfig] = {
    ITA_MODEL_ID: ModelConfig(ITA_MODEL_ID, "Italian", "ita", COUNSELOR_MODEL_ITA),
    ENG_MODEL_ID: ModelConfig(ENG_MODEL_ID, "English", "eng", COUNSELOR_MODEL_ENG),
}

if OWUI_PROCESS_MODEL in {"ITA", "IT"}:
    MODEL_CONFIGS: Dict[str, ModelConfig] = {ITA_MODEL_ID: ALL_MODEL_CONFIGS[ITA_MODEL_ID]}
elif OWUI_PROCESS_MODEL in {"ENG", "EN"}:
    MODEL_CONFIGS = {ENG_MODEL_ID: ALL_MODEL_CONFIGS[ENG_MODEL_ID]}
elif OWUI_PROCESS_MODEL in {"BOTH", "ALL", ""}:
    MODEL_CONFIGS = dict(ALL_MODEL_CONFIGS)
else:
    raise RuntimeError("OWUI_PROCESS_MODEL must be ITA, ENG, or BOTH")

MODEL_ALIASES: Dict[str, str] = {}
for mid in MODEL_CONFIGS:
    MODEL_ALIASES[mid] = mid
    MODEL_ALIASES[mid.removesuffix(":latest")] = mid

# Backward-compatible alias. Only add this alias if its target is exposed by the
# current process; otherwise /v1/models stays unambiguous for OpenWebUI.
legacy_target = os.getenv("OWUI_LEGACY_MODEL_TARGET", "ITA").strip().upper()
legacy_model = _ensure_latest_suffix(os.getenv("OWUI_MODEL_ID", "university-counselor:latest"))
legacy_canonical = ENG_MODEL_ID if legacy_target == "ENG" else ITA_MODEL_ID
if legacy_canonical in MODEL_CONFIGS:
    MODEL_ALIASES[legacy_model] = legacy_canonical
    MODEL_ALIASES[legacy_model.removesuffix(":latest")] = legacy_canonical


# -----------------------------------------------------------------------------
# OpenWebUI-compatible request models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any], Dict[str, Any]]
    name: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

    model_config = ConfigDict(extra="allow")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _now_ts() -> int:
    return int(time.time())


def _lang_name(dialog_language: str) -> str:
    return "Italian" if (dialog_language or "").strip().lower().startswith("it") else "English"


def _message_content_to_text(content: Union[str, List[Any], Dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p.strip()).strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return json.dumps(content, ensure_ascii=False)
    return str(content or "")


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _request_transcript(messages: List[ChatMessage]) -> List[Tuple[str, str]]:
    transcript: List[Tuple[str, str]] = []
    for msg in messages:
        if msg.role not in {"user", "assistant"}:
            continue
        text = _normalize_text(_message_content_to_text(msg.content))
        if text:
            transcript.append((msg.role, text))
    return transcript


def _user_only(transcript: List[Tuple[str, str]]) -> List[str]:
    return [text for role, text in transcript if role == "user"]


def _is_openwebui_task_message(text: str) -> bool:
    return (text or "").lstrip().startswith("### Task:")


def _first_nonempty_string(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _stable_session_key_from_request(request: Any, raw_request: Optional[Request] = None) -> Optional[str]:
    headers = raw_request.headers if raw_request is not None else {}
    header_value = _first_nonempty_string(
        headers.get("x-openwebui-chat-id") if headers else None,
        headers.get("x-chat-id") if headers else None,
        headers.get("x-conversation-id") if headers else None,
        headers.get("x-session-id") if headers else None,
    )
    if header_value:
        return header_value

    candidates: List[Any] = []
    for key in ("chat_id", "conversation_id", "session_id", "thread_id"):
        candidates.append(getattr(request, key, None))

    metadata = getattr(request, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("chat_id", "conversation_id", "session_id", "thread_id"):
            candidates.append(metadata.get(key))

    extra = getattr(request, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("chat_id", "conversation_id", "session_id", "thread_id"):
            candidates.append(extra.get(key))
        meta = extra.get("metadata")
        if isinstance(meta, dict):
            for key in ("chat_id", "conversation_id", "session_id", "thread_id"):
                candidates.append(meta.get(key))

    return _first_nonempty_string(*candidates)


def resolve_model(model: str) -> ModelConfig:
    key = (model or "").strip()
    canonical = MODEL_ALIASES.get(key)
    if not canonical:
        raise HTTPException(status_code=404, detail=f"Model not found: {model}")
    return MODEL_CONFIGS[canonical]


# -----------------------------------------------------------------------------
# Shared practice + retrievers
# -----------------------------------------------------------------------------

_SHARED_LOCK = threading.Lock()
_SHARED_PRACTICE: Optional[Dict[str, Any]] = None
_SHARED_RETRIEVERS: Dict[str, TimedRetriever] = {}


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _get_social_practice_path() -> str:
    return os.getenv(
        "SOCIAL_PRACTICE_PATH",
        str(_get_project_root() / "configuration_data" / "social_practices.json"),
    )


def get_shared_practice() -> Dict[str, Any]:
    global _SHARED_PRACTICE
    with _SHARED_LOCK:
        if _SHARED_PRACTICE is None:
            _SHARED_PRACTICE = get_social_practice(
                SOCIAL_PRACTICE_NAME,
                path=_get_social_practice_path(),
            )
        return _SHARED_PRACTICE


def _default_lancedb_dir(rag_lang: str) -> Path:
    dataset_dir = "RAG_University_Ita" if rag_lang == "ita" else "RAG_University_Eng"
    return _get_project_root() / dataset_dir / "Embeddings" / "lancedb"


def _default_embeddings_dir(rag_lang: str) -> Path:
    dataset_dir = "RAG_University_Ita" if rag_lang == "ita" else "RAG_University_Eng"
    return _get_project_root() / dataset_dir / "Embeddings"


def _env_for_lang(base: str, rag_lang: str) -> Optional[str]:
    specific = os.getenv(f"{base}_{rag_lang.upper()}", "").strip()
    if specific:
        return specific
    generic = os.getenv(base, "").strip()
    return generic or None


def get_shared_retriever(rag_lang: str) -> TimedRetriever:
    rag_lang = "ita" if rag_lang == "ita" else "eng"
    backend = os.getenv("SDIALOG_RAG_BACKEND", "lancedb").lower()
    cache_key = f"{backend}:{rag_lang}"

    with _SHARED_LOCK:
        if cache_key in _SHARED_RETRIEVERS:
            return _SHARED_RETRIEVERS[cache_key]

        _configure_rag_env()

        if backend == "lancedb":
            from RAG_Scripts.lancedb_university_retriever import LanceDBUniversityRetriever  # type: ignore

            table_name = os.getenv("SDIALOG_LANCEDB_TABLE", "universities")
            lancedb_dir = Path(_env_for_lang("SDIALOG_LANCEDB_DIR", rag_lang) or _default_lancedb_dir(rag_lang))
            if not (lancedb_dir / f"{table_name}.lance").exists():
                raise RuntimeError(
                    f"RAG LanceDB non trovato per lingua {rag_lang}: {lancedb_dir} table={table_name}. "
                    "Imposta SDIALOG_LANCEDB_DIR_ITA/ENG o ricostruisci l'indice."
                )
            logger.info("Using LanceDB index for %s: %s table=%s", rag_lang, lancedb_dir, table_name)
            retriever = TimedRetriever(
                LanceDBUniversityRetriever(
                    lang=rag_lang,
                    lancedb_dir=str(lancedb_dir),
                    table_name=table_name,
                )
            )
        else:
            from RAG_Scripts.rag_retriever import RAGRetriever  # type: ignore

            base = _default_embeddings_dir(rag_lang)
            faiss_path = _env_for_lang("SDIALOG_RAG_FAISS", rag_lang) or str(base / "rag.index.faiss")
            chunks_path = _env_for_lang("SDIALOG_RAG_CHUNKS", rag_lang) or str(base / "rag.chunks.jsonl")
            retriever = TimedRetriever(
                RAGRetriever(
                    lang=rag_lang,
                    faiss_path=faiss_path,
                    chunks_path=chunks_path,
                )
            )

        _SHARED_RETRIEVERS[cache_key] = retriever
        return retriever


# -----------------------------------------------------------------------------
# Per-session state
# -----------------------------------------------------------------------------

@dataclass
class RagGlobalsSnapshot:
    listener_patch_history: List[Dict[str, Any]] = field(default_factory=list)
    listener_patch_buffer: List[Dict[str, Any]] = field(default_factory=list)
    listener_patch_events: List[Dict[str, Any]] = field(default_factory=list)
    debug_snapshots: List[Any] = field(default_factory=list)
    last_rag_state: Any = None
    advice_options_shown: List[str] = field(default_factory=list)
    selected_option_event: Optional[Dict[str, Any]] = None
    current_listener_source_turn: Optional[int] = None


@dataclass
class CounselorSession:
    session_id: str
    model_id: str
    dialog_language: str
    rag_lang: str
    counselor_model: str
    agent: Agent
    session_key: Optional[str] = None
    transcript: List[Tuple[str, str]] = field(default_factory=list)
    rag_globals: RagGlobalsSnapshot = field(default_factory=RagGlobalsSnapshot)
    listener_mem: Dict[str, Any] = field(default_factory=rag_state.canonical_listener_memory)
    patch_count: int = 0
    last_printed_mem: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)


SESSIONS: Dict[str, CounselorSession] = {}
SESSIONS_BY_KEY: Dict[Tuple[str, str], str] = {}
SESSIONS_LOCK = threading.RLock()
LAST_CLEANUP_AT = 0.0


def _bounded_deepcopy(value: Any, max_items: int) -> Any:
    if isinstance(value, list) and max_items >= 0:
        if max_items == 0:
            return []
        return copy.deepcopy(value[-max_items:])
    return copy.deepcopy(value)


def _snapshot_rag_globals() -> RagGlobalsSnapshot:
    return RagGlobalsSnapshot(
        listener_patch_history=_bounded_deepcopy(getattr(rag_state, "_LISTENER_PATCH_HISTORY", []), MAX_PATCH_HISTORY),
        listener_patch_buffer=_bounded_deepcopy(getattr(rag_state, "_LISTENER_PATCH_BUFFER", []), MAX_PATCH_BUFFER),
        listener_patch_events=_bounded_deepcopy(getattr(rag_state, "_LISTENER_PATCH_EVENTS", []), MAX_PATCH_EVENTS),
        debug_snapshots=_bounded_deepcopy(getattr(rag_state, "_DEBUG_SNAPSHOTS", []), MAX_DEBUG_SNAPSHOTS),
        last_rag_state=copy.deepcopy(getattr(rag_state, "_LAST_RAG_STATE", None)),
        advice_options_shown=_bounded_deepcopy(getattr(rag_state, "_ADVICE_OPTIONS_SHOWN", []), MAX_ADVICE_OPTIONS),
        selected_option_event=copy.deepcopy(getattr(rag_state, "_SELECTED_OPTION_EVENT", None)),
        current_listener_source_turn=copy.deepcopy(getattr(rag_state, "_CURRENT_LISTENER_SOURCE_TURN", None)),
    )


def _restore_rag_globals(snapshot: RagGlobalsSnapshot) -> None:
    if hasattr(rag_state, "_LISTENER_PATCH_HISTORY"):
        rag_state._LISTENER_PATCH_HISTORY[:] = copy.deepcopy(snapshot.listener_patch_history)
    if hasattr(rag_state, "_LISTENER_PATCH_BUFFER"):
        rag_state._LISTENER_PATCH_BUFFER[:] = copy.deepcopy(snapshot.listener_patch_buffer)
    if hasattr(rag_state, "_LISTENER_PATCH_EVENTS"):
        rag_state._LISTENER_PATCH_EVENTS[:] = copy.deepcopy(snapshot.listener_patch_events)
    if hasattr(rag_state, "_DEBUG_SNAPSHOTS"):
        rag_state._DEBUG_SNAPSHOTS[:] = copy.deepcopy(snapshot.debug_snapshots)
    if hasattr(rag_state, "_LAST_RAG_STATE") and snapshot.last_rag_state is not None:
        rag_state._LAST_RAG_STATE = copy.deepcopy(snapshot.last_rag_state)
    if hasattr(rag_state, "_ADVICE_OPTIONS_SHOWN"):
        rag_state._ADVICE_OPTIONS_SHOWN[:] = copy.deepcopy(snapshot.advice_options_shown)
    if hasattr(rag_state, "_SELECTED_OPTION_EVENT"):
        rag_state._SELECTED_OPTION_EVENT = copy.deepcopy(snapshot.selected_option_event)
    if hasattr(rag_state, "_CURRENT_LISTENER_SOURCE_TURN"):
        rag_state._CURRENT_LISTENER_SOURCE_TURN = copy.deepcopy(snapshot.current_listener_source_turn)


def _make_session_postprocess(session: CounselorSession):
    def _postprocess(text: str) -> str:
        visible = sanitize_expert_output(text)

        patches = rag_state.get_listener_patches()
        new_patches = patches[session.patch_count:]
        if new_patches:
            for patch in new_patches:
                session.listener_mem = rag_state.apply_listener_patch(session.listener_mem, patch)
            session.patch_count = len(patches)

            mem_json = json.dumps(session.listener_mem, ensure_ascii=False, indent=2, sort_keys=True)
            if mem_json != session.last_printed_mem:
                print(f"\n[LISTENER MEMORY | session={session.session_id} | model={session.model_id}]", flush=True)
                print(mem_json, flush=True)
                session.last_printed_mem = mem_json
        return visible

    return _postprocess



# -----------------------------------------------------------------------------
# OpenAI-compatible strict chat-template compatibility
# -----------------------------------------------------------------------------

STRICT_SYSTEM_FIRST = os.getenv("SDIALOG_STRICT_SYSTEM_FIRST", "1").strip().lower() in {
    "1", "true", "yes", "on"
}


def _lc_message_role(message: Any) -> Optional[str]:
    """Return a normalized OpenAI-style role for LangChain/dict messages."""
    if isinstance(message, dict):
        role = message.get("role") or message.get("type")
    else:
        role = getattr(message, "role", None) or getattr(message, "type", None)
        if role is None:
            cls_name = message.__class__.__name__.lower()
            if "system" in cls_name:
                role = "system"
            elif "human" in cls_name:
                role = "human"
            elif "ai" in cls_name or "assistant" in cls_name:
                role = "ai"
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return str(role) if role is not None else None


def _lc_message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _make_lc_message(role: str, content: str) -> Any:
    """Create a LangChain chat message, falling back to dict if unavailable."""
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        if role == "system":
            return SystemMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        return HumanMessage(content=content)
    except Exception:
        return {"role": "assistant" if role == "assistant" else "system" if role == "system" else "user", "content": content}


def _strict_system_first_messages(messages: Any) -> Any:
    """
    Some OpenAI-compatible servers enforce chat templates where all system
    messages must be before the first user/assistant message. SDialog can insert
    additional system/context messages later in the memory. This rewrites late
    system messages into the next user message so strict backends do not reject
    the request with errors such as: Unexpected role 'system' after role
    'assistant'.
    """
    if not STRICT_SYSTEM_FIRST or not isinstance(messages, list):
        return messages

    fixed: List[Any] = []
    pending_late_system: List[str] = []
    seen_non_system = False
    changed = False

    for message in messages:
        role = _lc_message_role(message)
        content = _lc_message_content(message)

        if role == "system":
            if seen_non_system:
                if content.strip():
                    pending_late_system.append(content.strip())
                changed = True
                continue
            fixed.append(message)
            continue

        if role == "user" and pending_late_system:
            merged = (
                "[Additional system/context instructions]\n"
                + "\n\n".join(pending_late_system)
                + "\n\n[User message]\n"
                + content
            )
            fixed.append(_make_lc_message("user", merged))
            pending_late_system = []
            changed = True
        else:
            fixed.append(message)

        if role is not None:
            seen_non_system = True

    if pending_late_system:
        addition = "\n\n[Additional system/context instructions]\n" + "\n\n".join(pending_late_system)
        for idx in range(len(fixed) - 1, -1, -1):
            if _lc_message_role(fixed[idx]) == "user":
                fixed[idx] = _make_lc_message("user", _lc_message_content(fixed[idx]) + addition)
                changed = True
                break
        else:
            fixed.insert(0, _make_lc_message("system", "\n\n".join(pending_late_system)))
            changed = True

    if changed:
        logger.debug("Rewrote late system messages for strict OpenAI-compatible backend.")
    return fixed


def _patch_agent_strict_system_first(agent: Any) -> Any:
    if not STRICT_SYSTEM_FIRST:
        return agent
    original = getattr(agent, "_get_llm_response", None)
    if not callable(original):
        return agent

    def _wrapped_get_llm_response(messages: Any, *args: Any, **kwargs: Any) -> Any:
        return original(_strict_system_first_messages(messages), *args, **kwargs)

    setattr(agent, "_get_llm_response", _wrapped_get_llm_response)
    return agent


# -----------------------------------------------------------------------------
# Counselor factory
# -----------------------------------------------------------------------------


def build_session_counselor(session: CounselorSession) -> Agent:
    practice = get_shared_practice()
    rag_state.set_dialog_language(session.dialog_language)

    counselor_persona = Persona(
        name="University Counselor",
        age="middle-aged",
        gender="unspecified",
        role=practice.get("agent1_role", "UniversityCounselor"),
        background="Works at the university counseling office in Italy.",
        personality="",
        circumstances=f"Social practice: {practice.get('sp_name', 'University Counseling')}",
        rules="; ".join(practice.get("agent1_norms", []) or []),
        language=_lang_name(session.dialog_language),
    )

    counselor_agent = Agent(
        persona=counselor_persona,
        model=session.counselor_model,
        name=session.model_id,
        response_details=_build_expert_response_details(practice, session.dialog_language),
        think=ENABLE_AGENT_THINKING,
        postprocess_fn=_make_session_postprocess(session),
    )
    counselor_agent = _patch_agent_strict_system_first(counselor_agent)

    counselor_agent = counselor_agent | UniversityCounselorFlowOrchestrator(
        retriever=get_shared_retriever(session.rag_lang),
        required_slots=("academic_background", "field_of_interest", "region"),
        top_k=int(os.getenv("SDIALOG_COUNSELOR_TOP_K", "6")),
        candidate_pool_size=int(os.getenv("SDIALOG_CANDIDATE_POOL_SIZE", "4")),
        min_options_target=int(os.getenv("SDIALOG_MIN_OPTIONS_TARGET", "3")),
        final_top_n=int(os.getenv("SDIALOG_FINAL_TOP_N", "3")),
        max_ctx_chars=int(os.getenv("SDIALOG_MAX_CTX_CHARS", "2200")),
        history_turns=3,
        debug=os.getenv("SDIALOG_DEBUG", "0").lower() in {"1", "true", "yes"},
        dialog_language=session.dialog_language,
        practice=practice,
    )
    return counselor_agent


def create_new_session(
    config: ModelConfig,
    *,
    initial_transcript: Optional[List[Tuple[str, str]]] = None,
    session_key: Optional[str] = None,
) -> CounselorSession:
    sid = uuid.uuid4().hex[:12]
    session = CounselorSession(
        session_id=sid,
        model_id=config.model_id,
        dialog_language=config.dialog_language,
        rag_lang=config.rag_lang,
        counselor_model=config.counselor_model,
        agent=None,  # type: ignore[arg-type]
        session_key=session_key,
        transcript=list(initial_transcript or []),
    )

    with RAG_GLOBAL_LOCK:
        rag_state.set_dialog_language(session.dialog_language)
        rag_state.clear_listener_patches()
        rag_state.clear_debug_snapshots()
        if hasattr(rag_state, "clear_last_rag_state"):
            rag_state.clear_last_rag_state()
        session.agent = build_session_counselor(session)
        session.rag_globals = _snapshot_rag_globals()

    logger.info("Created counselor session %s model=%s lang=%s rag=%s backend=%s", sid, config.model_id, config.dialog_language, config.rag_lang, config.counselor_model)
    return session


def cleanup_old_sessions() -> None:
    global LAST_CLEANUP_AT
    if SESSION_TTL_SECONDS <= 0:
        return
    now = time.time()
    if now - LAST_CLEANUP_AT < SESSION_CLEANUP_INTERVAL_SECONDS:
        return
    LAST_CLEANUP_AT = now
    with SESSIONS_LOCK:
        expired = [sid for sid, session in SESSIONS.items() if now - session.last_seen_at > SESSION_TTL_SECONDS]
        for sid in expired:
            session = SESSIONS.pop(sid, None)
            if session and session.session_key:
                SESSIONS_BY_KEY.pop((session.model_id, session.session_key), None)
            logger.info("Removed expired session %s", sid)


def find_or_create_session(
    config: ModelConfig,
    messages: List[ChatMessage],
    *,
    session_key: Optional[str] = None,
) -> Tuple[CounselorSession, str]:
    cleanup_old_sessions()
    transcript = _request_transcript(messages)
    prefix = transcript[:-1] if transcript and transcript[-1][0] == "user" else transcript
    prefix_users = _user_only(prefix)

    with SESSIONS_LOCK:
        if session_key:
            sid = SESSIONS_BY_KEY.get((config.model_id, session_key))
            if sid:
                session = SESSIONS.get(sid)
                if session is not None:
                    session.last_seen_at = time.time()
                    return session, "existing-session-key"
                SESSIONS_BY_KEY.pop((config.model_id, session_key), None)

        for session in SESSIONS.values():
            if session.model_id == config.model_id and session.transcript == prefix:
                session.last_seen_at = time.time()
                return session, "existing"

        # Useful in OpenWebUI comparison mode: assistant messages can differ, but
        # user turns are stable.
        best_session: Optional[CounselorSession] = None
        best_score = -1.0
        for session in SESSIONS.values():
            if session.model_id != config.model_id:
                continue
            if _user_only(session.transcript) == prefix_users and session.last_seen_at > best_score:
                best_session = session
                best_score = session.last_seen_at
        if best_session is not None:
            best_session.last_seen_at = time.time()
            return best_session, "existing-user-only"

        session = create_new_session(config, initial_transcript=prefix, session_key=session_key)
        SESSIONS[session.session_id] = session
        if session_key:
            SESSIONS_BY_KEY[(config.model_id, session_key)] = session.session_id
        return session, "new"


# -----------------------------------------------------------------------------
# Agent execution
# -----------------------------------------------------------------------------


def _extract_content_from_events(events: List[Any]) -> str:
    if not events:
        return ""
    for event in reversed(events):
        if getattr(event, "action", None) == "utter":
            return str(getattr(event, "content", "") or "")
    return ""


def run_counselor_turn(session: CounselorSession, user_text: str) -> str:
    with session.lock:
        with RAG_GLOBAL_LOCK:
            _restore_rag_globals(session.rag_globals)
            # If the saved patch history was bounded, keep the postprocess cursor
            # in range so new patches created by this turn are still applied.
            try:
                session.patch_count = min(session.patch_count, len(rag_state.get_listener_patches()))
            except Exception:
                session.patch_count = 0
            rag_state.set_dialog_language(session.dialog_language)
            try:
                events = session.agent(user_text, return_events=True)
                content = _extract_content_from_events(events)
            finally:
                session.rag_globals = _snapshot_rag_globals()
        return content


def update_session_transcript(session: CounselorSession, request_messages: List[ChatMessage], user_text: str, assistant_text: str) -> None:
    transcript = _request_transcript(request_messages)
    prefix = transcript[:-1] if transcript and transcript[-1][0] == "user" else transcript
    session.transcript = prefix + [
        ("user", _normalize_text(user_text)),
        ("assistant", _normalize_text(assistant_text)),
    ]
    session.last_seen_at = time.time()


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(
    title="University Counselor Session Server",
    description="OpenWebUI-compatible API with one SDialog counselor instance per chat/session/model.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    with SESSIONS_LOCK:
        session_ids = list(SESSIONS.keys())
    return {
        "status": "ok",
        "models": list(MODEL_CONFIGS.keys()),
        "aliases": MODEL_ALIASES,
        "process_model": OWUI_PROCESS_MODEL,
        "pid": os.getpid(),
        "sessions": len(session_ids),
        "indexed_sessions": len(SESSIONS_BY_KEY),
        "session_ids": session_ids,
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": _now_ts(), "owned_by": "sdialog"}
            for mid in MODEL_CONFIGS.keys()
        ],
    }


def _chat_completion_payload(request: ChatCompletionRequest, content: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": _now_ts(),
        "model": request.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream_openai_chunks(request: ChatCompletionRequest, content: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = _now_ts()
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
    for match in re.finditer(r"\S+\s*", content or ""):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": match.group(0)}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    config = resolve_model(request.model)
    request.model = config.model_id

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found.")

    last_user_text = _message_content_to_text(user_messages[-1].content).strip()
    if _is_openwebui_task_message(last_user_text):
        content = ""
        if request.stream:
            return StreamingResponse(_stream_openai_chunks(request, content), media_type="text/event-stream")
        return JSONResponse(content=_chat_completion_payload(request, content))

    session_key = _stable_session_key_from_request(request, raw_request)
    session, status = find_or_create_session(config, request.messages, session_key=session_key)
    logger.info(
        "POST /v1/chat/completions model=%s lang=%s rag=%s session=%s status=%s messages=%d stream=%s",
        request.model,
        config.dialog_language,
        config.rag_lang,
        session.session_id,
        status,
        len(request.messages),
        bool(request.stream),
    )

    try:
        assistant_text = run_counselor_turn(session, last_user_text)
        update_session_transcript(session, request.messages, last_user_text, assistant_text)
    except Exception as exc:
        logger.exception("Error while processing session %s", session.session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if request.stream:
        return StreamingResponse(_stream_openai_chunks(request, assistant_text), media_type="text/event-stream")
    return JSONResponse(content=_chat_completion_payload(request, assistant_text))


@app.get("/api/tags")
def ollama_tags() -> Dict[str, Any]:
    return {
        "models": [
            {
                "name": mid,
                "model": mid,
                "modified_at": datetime.now().isoformat() + "Z",
                "size": 0,
                "digest": f"sha256:{'0' * 64}",
                "details": {
                    "format": "sdialog",
                    "family": "sdialog",
                    "families": ["sdialog"],
                    "parameter_size": "unknown",
                    "quantization_level": "none",
                },
            }
            for mid in MODEL_CONFIGS.keys()
        ]
    }


@app.get("/api/version")
def ollama_version() -> Dict[str, str]:
    return {"version": "0.2.0-sdialog-session-server"}


def main() -> None:
    logger.info("Starting session server on %s:%s", HOST, PORT)
    logger.info("Models: %s", ", ".join(MODEL_CONFIGS.keys()))
    logger.info("Aliases: %s", MODEL_ALIASES)
    logger.info("Process model mode: %s", OWUI_PROCESS_MODEL)
    logger.info("OpenAI-compatible backend base: %s", SELECTED_OPENAI_API_BASE)
    logger.info("Agent thinking enabled: %s", ENABLE_AGENT_THINKING)
    for mid, cfg in MODEL_CONFIGS.items():
        logger.info("Counselor backend for %s: %s", mid, cfg.counselor_model)
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"), workers=1)


if __name__ == "__main__":
    main()
