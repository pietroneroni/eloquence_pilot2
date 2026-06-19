# university_counseling/owui_session_server.py
"""
Open WebUI-compatible session server for the university counselor.

Why this file exists
--------------------
`agent.serve(...)` exposes one Agent instance. That is fine for one active chat,
but your UniversityCounselorFlowOrchestrator is stateful, so multiple Open WebUI
chats can share the same flow state by accident.

This server exposes an OpenAI-compatible /v1/chat/completions endpoint and keeps
one independent counselor Agent instance per Open WebUI chat transcript.

Run from the project root with:

    python -m university_counseling.owui_session_server

Configure Open WebUI with:

    Base URL: http://host.docker.internal:1333/v1
    API Key: EMPTY
    Model ID: university-counselor:latest
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

import uvicorn
import sdialog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

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

# By default this script exposes two OpenWebUI-visible models, one with a
# forced male listener memory and one with a forced female listener memory.
# For backward compatibility, setting OWUI_FORCED_GENDER=male|female exposes
# a single model named OWUI_MODEL_ID.
BASE_MODEL_ID = os.getenv("OWUI_MODEL_ID", "university-counselor")
FORCED_GENDER = os.getenv("OWUI_FORCED_GENDER", "").strip().lower()

if FORCED_GENDER:
    if FORCED_GENDER not in {"male", "female"}:
        raise ValueError("OWUI_FORCED_GENDER must be 'male' or 'female'.")

    MODEL_ID = BASE_MODEL_ID if BASE_MODEL_ID.endswith(":latest") else f"{BASE_MODEL_ID}:latest"
    MODEL_ALIASES = {
        MODEL_ID: (MODEL_ID, FORCED_GENDER),
        MODEL_ID.removesuffix(":latest"): (MODEL_ID, FORCED_GENDER),
    }
    MODEL_IDS = [MODEL_ID]
else:
    MODEL_ID = BASE_MODEL_ID
    MALE_MODEL_ID = os.getenv("OWUI_MALE_MODEL_ID", f"{BASE_MODEL_ID}-male:latest")
    FEMALE_MODEL_ID = os.getenv("OWUI_FEMALE_MODEL_ID", f"{BASE_MODEL_ID}-female:latest")
    MODEL_ALIASES = {
        MALE_MODEL_ID: (MALE_MODEL_ID, "male"),
        MALE_MODEL_ID.removesuffix(":latest"): (MALE_MODEL_ID, "male"),
        FEMALE_MODEL_ID: (FEMALE_MODEL_ID, "female"),
        FEMALE_MODEL_ID.removesuffix(":latest"): (FEMALE_MODEL_ID, "female"),
    }
    MODEL_IDS = [MALE_MODEL_ID, FEMALE_MODEL_ID]
PORT = int(os.getenv("OWUI_AGENT_PORT", "1333"))
HOST = os.getenv("OWUI_AGENT_HOST", "0.0.0.0")

DIALOG_LANGUAGE = os.getenv("DIALOG_LANGUAGE", "Italian")
SOCIAL_PRACTICE_NAME = os.getenv("SOCIAL_PRACTICE_NAME", "university_counseling")
# OpenAI-compatible backend used by SDialog to call the real LLM.
# This server exposes the counselor to OpenWebUI, but it still needs a separate
# LLM backend reachable at OPENAI_API_BASE/OPENAI_BASE_URL.
os.environ.setdefault("OPENAI_API_KEY", os.getenv("SDIALOG_OPENAI_API_KEY", "kk"))
OPENAI_API_BASE = os.getenv(
    "SDIALOG_OPENAI_API_BASE",
    os.getenv("OPENAI_API_BASE", "http://127.0.0.1:10007/v1"),
)
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

COUNSELOR_MODEL = os.getenv(
    "COUNSELOR_MODEL",
    os.getenv("SDIALOG_MODEL_URI", "openai:gemma-3-27b-it-q8_0"),
)
sdialog.config.llm(COUNSELOR_MODEL)

# Session cleanup. Increase this if you keep many chats open for a long time.
SESSION_TTL_SECONDS = int(os.getenv("OWUI_SESSION_TTL_SECONDS", str(60 * 60 * 6)))

# SDialog/AgentWithRag contains a few module-level globals used during generation
# and postprocessing. This lock ensures two sessions do not mutate them at the
# same time.
RAG_GLOBAL_LOCK = threading.RLock()

logging.basicConfig(
    level=os.getenv("OWUI_SESSION_LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("owui-session-server")


# -----------------------------------------------------------------------------
# OpenAI-compatible request models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any], Dict[str, Any]]
    name: Optional[str] = None

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

    class Config:
        extra = "allow"


# -----------------------------------------------------------------------------
# Helpers: text, transcripts, Open WebUI task filtering
# -----------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _now_ts() -> int:
    return int(time.time())


def _lang_name(dialog_language: str) -> str:
    return "Italian" if (dialog_language or "").strip().lower().startswith("it") else "English"


def resolve_model(model: str) -> Tuple[str, str]:
    """Return canonical model id and the forced listener gender for that model."""
    key = (model or "").strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    raise HTTPException(status_code=404, detail=f"Model not found: {model}")


def _force_gender_in_memory(memory: Dict[str, Any], forced_gender: str) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        memory = rag_state.canonical_listener_memory()
    inferred = memory.setdefault("inferred", {})
    if not isinstance(inferred, dict):
        memory["inferred"] = {}
        inferred = memory["inferred"]
    inferred["gender"] = forced_gender
    return memory


def _force_gender_in_patch(patch: Dict[str, Any], forced_gender: str) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        patch = {}
    inferred = patch.setdefault("inferred", {})
    if not isinstance(inferred, dict):
        patch["inferred"] = {}
        inferred = patch["inferred"]
    inferred["gender"] = forced_gender
    return patch


def _force_gender_in_global_listener_patches(forced_gender: str) -> None:
    """
    sanitize_expert_output extracts <LISTENER_PATCH> blocks into module-level
    patch history/buffer. We rewrite those patches before the orchestrator later
    consumes them, so the counselor's own listener memory also sees the forced
    gender, not the model-inferred gender.
    """
    history = getattr(rag_state, "_LISTENER_PATCH_HISTORY", None)
    if isinstance(history, list):
        for i, patch in enumerate(history):
            history[i] = _force_gender_in_patch(patch, forced_gender)

    buffer = getattr(rag_state, "_LISTENER_PATCH_BUFFER", None)
    if isinstance(buffer, list):
        for i, patch in enumerate(buffer):
            buffer[i] = _force_gender_in_patch(patch, forced_gender)


def _message_content_to_text(content: Union[str, List[Any], Dict[str, Any]]) -> str:
    """
    OpenAI-compatible clients usually send string content, but multimodal clients
    may send lists/dicts. We reduce non-string content to readable text so the
    counselor gets a normal user utterance.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                # Common OpenAI multimodal format: {"type":"text", "text":"..."}
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
    """
    Transcript used to identify Open WebUI chats. System messages are ignored
    because they can change independently of the actual conversation.
    """
    transcript: List[Tuple[str, str]] = []
    for msg in messages:
        if msg.role not in {"user", "assistant"}:
            continue
        text = _normalize_text(_message_content_to_text(msg.content))
        if text:
            transcript.append((msg.role, text))
    return transcript

def _user_only(transcript: List[Tuple[str, str]]) -> List[str]:
    """
    Return only user turns from a transcript.

    In Open WebUI comparison mode, assistant messages can differ from what
    the server stored, but user messages are usually stable. This helps recover
    the correct session instead of creating a new one.
    """
    return [text for role, text in transcript if role == "user"]

def _is_openwebui_task_message(text: str) -> bool:
    """
    SDialog's own server ignores Open WebUI task messages that start with
    '### Task:'. We do the same so title/tag/autocomplete calls do not advance
    the stateful counselor flow.
    """
    return (text or "").lstrip().startswith("### Task:")


# -----------------------------------------------------------------------------
# Shared project resources: practice + retriever
# -----------------------------------------------------------------------------

_SHARED_LOCK = threading.Lock()
_SHARED_PRACTICE: Optional[Dict[str, Any]] = None
_SHARED_RETRIEVER: Optional[TimedRetriever] = None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_lancedb_dir() -> Path:
    return _get_project_root() / "RAG_University_Eng" / "Embeddings" / "lancedb"


def _ensure_lancedb_env() -> None:
    """
    Keep the RAG environment aligned with the project layout used on BSC.

    _configure_rag_env() can disable RAG when it does not find its default path.
    In this project the LanceDB index is under RAG_University_Eng/Embeddings/lancedb,
    so we re-enable RAG if the configured or default LanceDB table exists.
    """
    table_name = os.environ.get("SDIALOG_LANCEDB_TABLE", "universities")
    configured_dir = os.environ.get("SDIALOG_LANCEDB_DIR")
    lancedb_dir = Path(configured_dir) if configured_dir else _default_lancedb_dir()

    if (lancedb_dir / f"{table_name}.lance").exists():
        os.environ["SDIALOG_ENABLE_RAG"] = "1"
        os.environ["SDIALOG_RAG_BACKEND"] = "lancedb"
        os.environ["SDIALOG_LANCEDB_DIR"] = str(lancedb_dir)
        os.environ["SDIALOG_LANCEDB_TABLE"] = table_name
        logger.info("Using LanceDB index: %s table=%s", lancedb_dir, table_name)



def _get_social_practice_path() -> str:
    project_root = _get_project_root()
    return os.getenv(
        "SOCIAL_PRACTICE_PATH",
        str(project_root / "configuration_data" / "social_practices.json"),
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


def get_shared_retriever() -> TimedRetriever:
    global _SHARED_RETRIEVER

    with _SHARED_LOCK:
        if _SHARED_RETRIEVER is not None:
            return _SHARED_RETRIEVER

        _configure_rag_env()
        _ensure_lancedb_env()

        if os.environ.get("SDIALOG_ENABLE_RAG", "0") != "1":
            backend = os.environ.get("SDIALOG_RAG_BACKEND", "lancedb").lower()

            if backend == "lancedb":
                raise RuntimeError(
                    "RAG non abilitato: non trovo l'indice LanceDB. "
                    "Esegui prima: python RAG_Scripts/Build_lancedb_index_university.py "
                    "e controlla il path RAG_University_Eng/Embeddings/lancedb."
                )

            raise RuntimeError(
                "RAG non abilitato: non trovo rag.index.faiss e rag.chunks.jsonl. "
                "Ricostruisci l'indice FAISS o controlla i path in RAG_University_Eng/Embeddings."
            )

        backend = os.environ.get("SDIALOG_RAG_BACKEND", "lancedb").lower()

        if backend == "lancedb":
            from RAG_Scripts.lancedb_university_retriever import LanceDBUniversityRetriever  # type: ignore

            _SHARED_RETRIEVER = TimedRetriever(
                LanceDBUniversityRetriever(
                    lancedb_dir=os.environ.get(
                        "SDIALOG_LANCEDB_DIR",
                        str(_get_project_root() / "RAG_University_Eng" / "Embeddings" / "lancedb"),
                    ),
                    table_name=os.environ.get("SDIALOG_LANCEDB_TABLE", "universities"),
                )
            )
        else:
            from RAG_Scripts.rag_retriever import RAGRetriever  # type: ignore

            _SHARED_RETRIEVER = TimedRetriever(
                RAGRetriever(
                    faiss_path=os.environ.get("SDIALOG_RAG_FAISS", ""),
                    chunks_path=os.environ.get("SDIALOG_RAG_CHUNKS", ""),
                )
            )

        return _SHARED_RETRIEVER

# -----------------------------------------------------------------------------
# Per-session listener logging and global state snapshots
# -----------------------------------------------------------------------------

@dataclass
class RagGlobalsSnapshot:
    listener_patch_history: List[Dict[str, Any]] = field(default_factory=list)
    listener_patch_buffer: List[Dict[str, Any]] = field(default_factory=list)
    debug_snapshots: List[Any] = field(default_factory=list)
    last_rag_state: Any = None


@dataclass
class CounselorSession:
    session_id: str
    model_id: str
    forced_gender: str
    agent: Agent
    transcript: List[Tuple[str, str]] = field(default_factory=list)
    rag_globals: RagGlobalsSnapshot = field(default_factory=RagGlobalsSnapshot)
    listener_mem: Dict[str, Any] = field(default_factory=rag_state.canonical_listener_memory)
    patch_count: int = 0
    last_printed_mem: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)


SESSIONS: Dict[str, CounselorSession] = {}
SESSIONS_LOCK = threading.RLock()


def _snapshot_rag_globals() -> RagGlobalsSnapshot:
    return RagGlobalsSnapshot(
        listener_patch_history=copy.deepcopy(getattr(rag_state, "_LISTENER_PATCH_HISTORY", [])),
        listener_patch_buffer=copy.deepcopy(getattr(rag_state, "_LISTENER_PATCH_BUFFER", [])),
        debug_snapshots=copy.deepcopy(getattr(rag_state, "_DEBUG_SNAPSHOTS", [])),
        last_rag_state=copy.deepcopy(getattr(rag_state, "_LAST_RAG_STATE", None)),
    )


def _restore_rag_globals(snapshot: RagGlobalsSnapshot) -> None:
    # These names are module-level globals in AgentWithRag.py.
    if hasattr(rag_state, "_LISTENER_PATCH_HISTORY"):
        rag_state._LISTENER_PATCH_HISTORY[:] = copy.deepcopy(snapshot.listener_patch_history)
    if hasattr(rag_state, "_LISTENER_PATCH_BUFFER"):
        rag_state._LISTENER_PATCH_BUFFER[:] = copy.deepcopy(snapshot.listener_patch_buffer)
    if hasattr(rag_state, "_DEBUG_SNAPSHOTS"):
        rag_state._DEBUG_SNAPSHOTS[:] = copy.deepcopy(snapshot.debug_snapshots)
    if hasattr(rag_state, "_LAST_RAG_STATE") and snapshot.last_rag_state is not None:
        rag_state._LAST_RAG_STATE = copy.deepcopy(snapshot.last_rag_state)


def _make_session_postprocess(session: CounselorSession):
    """
    Build a postprocess function bound to one session. It preserves your existing
    sanitize_expert_output behavior and prints only this session's listener memory
    when the memory actually changes.
    """

    def _postprocess(text: str) -> str:
        visible = sanitize_expert_output(text)

        _force_gender_in_global_listener_patches(session.forced_gender)

        patches = rag_state.get_listener_patches()
        new_patches = patches[session.patch_count:]

        if new_patches:
            for patch in new_patches:
                patch = _force_gender_in_patch(patch, session.forced_gender)
                session.listener_mem = rag_state.apply_listener_patch(session.listener_mem, patch)
                session.listener_mem = _force_gender_in_memory(session.listener_mem, session.forced_gender)

            session.patch_count = len(patches)

            mem_json = json.dumps(
                session.listener_mem,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

            if mem_json != session.last_printed_mem:
                print(f"\n[LISTENER MEMORY | session={session.session_id}]", flush=True)
                print(mem_json, flush=True)
                session.last_printed_mem = mem_json

        return visible

    return _postprocess


# -----------------------------------------------------------------------------
# Counselor factory
# -----------------------------------------------------------------------------


def build_session_counselor(session: CounselorSession) -> Agent:
    practice = get_shared_practice()

    rag_state.set_dialog_language(DIALOG_LANGUAGE)

    counselor_persona = Persona(
        name="University Counselor",
        age="middle-aged",
        gender="unspecified",
        role=practice.get("agent1_role", "UniversityCounselor"),
        background="Works at the university counseling office in Italy.",
        personality="",
        circumstances=f"Social practice: {practice.get('sp_name', 'University Counseling')}",
        rules="; ".join(practice.get("agent1_norms", []) or []),
        language=_lang_name(DIALOG_LANGUAGE),
    )

    response_details = _build_expert_response_details(practice, DIALOG_LANGUAGE)
    response_details += (
        "\nHIDDEN LISTENER OVERRIDE (STRICT): "
        f"For the listener field inferred.gender, always use {session.forced_gender!r}. "
        "Do not use any other gender value even if the student's wording suggests it."
    )

    counselor_agent = Agent(
        persona=counselor_persona,
        model=COUNSELOR_MODEL,
        name=session.model_id,
        response_details=response_details,
        think=True,
        postprocess_fn=_make_session_postprocess(session),
    )

    counselor_agent = counselor_agent | UniversityCounselorFlowOrchestrator(
        retriever=get_shared_retriever(),
        required_slots=("academic_background", "field_of_interest", "region"),
        top_k=12,
        candidate_pool_size=8,
        max_ctx_chars=2200,
        history_turns=3,
        debug=False,
        dialog_language=DIALOG_LANGUAGE,
        practice=practice,
    )

    return counselor_agent


def create_new_session(
    *,
    model_id: str,
    forced_gender: str,
    initial_transcript: Optional[List[Tuple[str, str]]] = None,
) -> CounselorSession:
    sid = uuid.uuid4().hex[:12]

    # Create an empty shell first so the postprocess closure can refer to it.
    session = CounselorSession(
        session_id=sid,
        model_id=model_id,
        forced_gender=forced_gender,
        agent=None,  # type: ignore[arg-type]
        transcript=list(initial_transcript or []),
    )
    session.listener_mem = _force_gender_in_memory(session.listener_mem, forced_gender)

    # Avoid contaminating a new session with global listener patches left by a
    # previous generation.
    with RAG_GLOBAL_LOCK:
        rag_state.clear_listener_patches()
        rag_state.clear_debug_snapshots()
        if hasattr(rag_state, "clear_last_rag_state"):
            rag_state.clear_last_rag_state()

        session.agent = build_session_counselor(session)
        session.rag_globals = _snapshot_rag_globals()

    logger.info("Created new counselor session %s model=%s forced_gender=%s", sid, model_id, forced_gender)
    return session


def cleanup_old_sessions() -> None:
    if SESSION_TTL_SECONDS <= 0:
        return

    now = time.time()
    with SESSIONS_LOCK:
        expired = [
            sid for sid, session in SESSIONS.items()
            if now - session.last_seen_at > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del SESSIONS[sid]
            logger.info("Removed expired session %s", sid)


def find_or_create_session(
    *,
    model_id: str,
    forced_gender: str,
    messages: List[ChatMessage],
) -> Tuple[CounselorSession, str]:
    """
    Open WebUI's standard OpenAI-compatible request usually does not include a
    stable chat_id for custom providers.

    First we try exact transcript matching. If that fails, we fallback to
    matching only the user turns. This is more robust in Open WebUI comparison
    mode, where assistant-side history can differ from what this server stored.
    """
    cleanup_old_sessions()

    transcript = _request_transcript(messages)
    prefix = transcript[:-1] if transcript and transcript[-1][0] == "user" else transcript
    prefix_users = _user_only(prefix)

    with SESSIONS_LOCK:
        # 1) Exact match: normal path.
        for session in SESSIONS.values():
            if session.model_id == model_id and session.transcript == prefix:
                session.last_seen_at = time.time()
                return session, "existing"

        # 2) Relaxed match: same model + same user-turn history.
        # This fixes Open WebUI comparison mode when assistant messages differ.
        best_session: Optional[CounselorSession] = None
        best_score = -1.0

        for session in SESSIONS.values():
            if session.model_id != model_id:
                continue

            session_users = _user_only(session.transcript)

            if session_users == prefix_users:
                # Prefer the most recently used matching session.
                score = session.last_seen_at
                if score > best_score:
                    best_score = score
                    best_session = session

        if best_session is not None:
            best_session.last_seen_at = time.time()
            logger.info(
                "Recovered existing session by user-only transcript match: session=%s model=%s",
                best_session.session_id,
                model_id,
            )
            return best_session, "existing-user-only"

        # 3) Brand-new chat.
        session = create_new_session(
            model_id=model_id,
            forced_gender=forced_gender,
            initial_transcript=prefix,
        )
        SESSIONS[session.session_id] = session
        return session, "new"

# -----------------------------------------------------------------------------
# Agent execution
# -----------------------------------------------------------------------------


def _extract_content_from_events(events: List[Any]) -> str:
    if not events:
        return ""

    # Prefer the last utterance. Some event lists can contain thinking/tool events.
    for event in reversed(events):
        if getattr(event, "action", None) == "utter":
            return str(getattr(event, "content", "") or "")

    return ""


def run_counselor_turn(session: CounselorSession, user_text: str) -> str:
    with session.lock:
        with RAG_GLOBAL_LOCK:
            _restore_rag_globals(session.rag_globals)

            try:
                events = session.agent(user_text, return_events=True)
                content = _extract_content_from_events(events)
            finally:
                # Save global AgentWithRag state back into this session even if
                # generation raises, so debugging state is not lost.
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
    description="OpenAI-compatible API with one SDialog counselor instance per chat/session.",
    version="1.0.0",
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
        "models": MODEL_IDS,
        "sessions": len(session_ids),
        "session_ids": session_ids,
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": _now_ts(),
                "owned_by": "sdialog",
            }
            for model_id in MODEL_IDS
        ],
    }


def _chat_completion_payload(request: ChatCompletionRequest, content: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": _now_ts(),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _stream_openai_chunks(request: ChatCompletionRequest, content: str):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = _now_ts()

    # First chunk declares assistant role.
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

    # Stream word-ish chunks to keep Open WebUI responsive.
    for match in re.finditer(r"\S+\s*", content or ""):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": match.group(0)},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    model_id, forced_gender = resolve_model(request.model)
    request.model = model_id

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found.")

    last_user_text = _message_content_to_text(user_messages[-1].content).strip()

    # Do not let Open WebUI title/tag/task calls advance the counselor flow.
    if _is_openwebui_task_message(last_user_text):
        content = ""
        if request.stream:
            return StreamingResponse(
                _stream_openai_chunks(request, content),
                media_type="text/event-stream",
            )
        return JSONResponse(content=_chat_completion_payload(request, content))

    session, status = find_or_create_session(
        model_id=model_id,
        forced_gender=forced_gender,
        messages=request.messages,
    )
    logger.info(
        "POST /v1/chat/completions model=%s forced_gender=%s session=%s status=%s messages=%d stream=%s",
        request.model,
        forced_gender,
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
        return StreamingResponse(
            _stream_openai_chunks(request, assistant_text),
            media_type="text/event-stream",
        )

    return JSONResponse(content=_chat_completion_payload(request, assistant_text))


# Optional Ollama-compatible discovery endpoints. Some UIs probe these.
@app.get("/api/tags")
def ollama_tags() -> Dict[str, Any]:
    return {
        "models": [
            {
                "name": model_id,
                "model": model_id,
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
            for model_id in MODEL_IDS
        ]
    }


@app.get("/api/version")
def ollama_version() -> Dict[str, str]:
    return {"version": "0.1.0-sdialog-session-server"}


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    logger.info("Starting session server on %s:%s", HOST, PORT)
    logger.info("Model IDs: %s", ", ".join(MODEL_IDS))
    logger.info("Counselor model backend: %s", COUNSELOR_MODEL)
    logger.info("OpenAI-compatible LLM base URL: %s", OPENAI_API_BASE)
    logger.info("Dialog language: %s", DIALOG_LANGUAGE)
    logger.info("SESSION MATCHING: exact + user-only fallback enabled")
    logger.info("FORCED_GENDER: %s", FORCED_GENDER)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )

if __name__ == "__main__":
    main()
