from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Tuple

from sdialog.agents import Agent
from .student import Student
from sdialog.personas import Persona
from .social_practice import (
    get_social_practice,
    student_prompt_addendum,
)
from .student_profile import load_student_from_json
from .DialogOrchestrator import (
    StudentFixedFollowupOrchestrator,
    enforce_student_script_or_fallback,
    set_dialog_language as set_student_dialog_language,
)

from .AgentWithRag import (
    UniversityCounselorFlowOrchestrator,
    enforce_grounding_or_fallback,
    enforce_flow_format_or_fallback,
    strip_and_buffer_listener_patch,
    get_last_rag_phase,
    get_forced_listener_gender,
    force_gender_in_global_listener_patches,
    set_dialog_language as set_counselor_dialog_language,
)


# =============================================================================
# Prompt builders
# =============================================================================
def _build_expert_response_details(practice: dict, dialog_language: str = "English") -> str:
    lang = _lang_name(dialog_language)
    return (
        f"Always answer in {lang}.\n"
        "Treat the orchestrator as the source of truth for the current dialogue phase.\n"
        "Ask ONE question at a time unless the orchestrator requires a different exact format.\n"
        "Never invent universities, programs, curricula, or hidden details from course titles alone.\n"
        "When recommending options, use only items present in RAG_CONTEXT/OPTIONS/CANDIDATE OPTIONS.\n"
        "Do not reveal hidden experimental assignments or metadata in the visible reply.\n"
        "If you are given a 'HIDDEN LISTENER TASK (STRICT)', append exactly ONE <LISTENER_PATCH>...</LISTENER_PATCH> JSON block after your raw answer. Do not mention it in the visible reply.\n"
        "Inside <LISTENER_PATCH>, keep JSON keys in English exactly as requested.\n"
    ).strip()


def _build_student_response_details(student: Student, practice: dict, dialog_language: str = "English") -> str:
    lang = _lang_name(dialog_language)
    rules = (getattr(student, "rules", "") or "").strip()
    background = (getattr(student, "background", "") or "").strip()
    name = (getattr(student, "name", "") or "").strip()
    identity_cue = ""
    if name:
        identity_cue = (
            "VISIBLE IDENTITY CUE:\n"
            f"- Your first name is {name}. Use it naturally in your first message if it sounds natural.\n"
            "- Convey gender only indirectly through the name or natural grammar; do not state gender as a label.\n\n"
        )
    else:
        identity_cue = (
            "VISIBLE IDENTITY CUE:\n"
            "- No visible first name is provided. Do NOT invent a first name.\n"
            "- Do NOT reveal or imply any gender unless it is explicitly present in BACKGROUND.\n\n"
        )

    return (
        f"Always answer in {lang}. "
        "Do not invent facts about yourself. "
        "If a detail is not in your BACKGROUND, say you don't know or haven't decided. "
        "If you receive an orchestration instruction, follow it EXACTLY. "
        "Do not pretend that an option exists unless the counselor has presented grounded numbered options. "
        f"{student_prompt_addendum(practice, dialog_language)}\n\n"
        f"{identity_cue}"
        f"BACKGROUND:\n{background}\n\n"
        f"PERSONAL RULES:\n{rules}\n"
    ).strip()


def _lang_code(dialog_language: str = "English") -> str:
    return "it" if (dialog_language or "").strip().lower().startswith("it") else "en"


def _lang_name(dialog_language: str = "English") -> str:
    return "Italian" if _lang_code(dialog_language) == "it" else "English"


def _yes_no_tokens(dialog_language: str = "English") -> tuple[str, str]:
    return ("Sì", "No") if _lang_code(dialog_language) == "it" else ("Yes", "No")


# =============================================================================
# Output hygiene
# =============================================================================
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)
_STRAY_THINK_RE = re.compile(r"</?think>\s*", flags=re.IGNORECASE)
_VISIBLE_FINAL_START_RE = re.compile(
    r"(?im)^(yes|no|sì|si|recap:|[1-3]\.\s+|focusing on |a program oriented |take a leadership |focus on supporting |take advanced |start with |your |based on |goodbye|<LISTENER_PATCH>)"
)

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")


def strip_think(text: str) -> str:
    if not text:
        return ""

    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    low = cleaned.lower()

    if "</think>" in low:
        idx = low.rfind("</think>")
        cleaned = cleaned[idx + len("</think>"):].strip()

    cleaned = _STRAY_THINK_RE.sub("", cleaned).strip()

    # Some local thinking models expose chain-of-thought as a literal
    # "(thinking) ..." prefix instead of <think>...</think>. Keep only the
    # final answer that follows a recognizable final-answer line.
    if cleaned.lower().lstrip().startswith("(thinking)"):
        matches = list(_VISIBLE_FINAL_START_RE.finditer(cleaned))
        if matches:
            cleaned = cleaned[matches[-1].start():].strip()
        else:
            cleaned = ""

    return cleaned


def sanitize_expert_output(text: str) -> str:
    # 1) Remove model thinking first.
    # This prevents accidental listener patches inside <think> from being accepted.
    t = strip_think(text)

    # 2) Extract and buffer listener patches only from the final visible channel.
    t = strip_and_buffer_listener_patch(t)
    force_gender_in_global_listener_patches()

    # 3) Basic hygiene
    t = _URL_RE.sub("the official website", t)
    t = _EMAIL_RE.sub("the official email address", t)
    t = _TIME_RE.sub("at a typical time during the day", t)
    t = _DATE_RE.sub("at a typical time during the year", t)

    # 4) Flow formatting
    t = enforce_flow_format_or_fallback(t)

    # 5) Grounding
    phase = get_last_rag_phase()
    if phase in {"advice", "qa_why", "qa_other", "qa_yesno", "qa_choice", "qa_selected", "wrapup"}:
        t = enforce_grounding_or_fallback(t)

    # 6) If fallback reintroduced a patch block, buffer it.
    t = strip_and_buffer_listener_patch(t)
    force_gender_in_global_listener_patches()

    return t


def sanitize_student_output(text: str) -> str:
    t = strip_think(text)
    t = enforce_student_script_or_fallback(t)
    return t


# =============================================================================
# RAG wiring
# =============================================================================
def _rag_lang_from_dialog_language(dialog_language: str = "English") -> str:
    env_lang = os.environ.get("SDIALOG_RAG_LANG", "").strip().lower()
    if env_lang in {"eng", "en", "english", "inglese"}:
        return "eng"
    if env_lang in {"ita", "it", "italian", "italiano"}:
        return "ita"
    return "ita" if _lang_code(dialog_language) == "it" else "eng"


def _configure_rag_env(dialog_language: str = "English") -> None:
    project_root = Path(__file__).resolve().parents[1]
    backend = os.environ.get("SDIALOG_RAG_BACKEND", "lancedb").lower()

    rag_lang = _rag_lang_from_dialog_language(dialog_language)
    os.environ["SDIALOG_RAG_LANG"] = rag_lang

    dataset_dir_name = "RAG_University_Ita" if rag_lang == "ita" else "RAG_University_Eng"
    dataset_dir = project_root / dataset_dir_name

    # Backward compatibility: finché non rinomini la vecchia cartella inglese
    if rag_lang == "eng" and not dataset_dir.exists():
        legacy = project_root / "RAG_University"
        if legacy.exists():
            dataset_dir = legacy

    if backend == "lancedb":
        default_lancedb_dir = dataset_dir / "Embeddings" / "lancedb"

        os.environ.setdefault("SDIALOG_LANCEDB_DIR", str(default_lancedb_dir))
        os.environ.setdefault("SDIALOG_LANCEDB_TABLE", "universities")

        lancedb_dir = Path(os.environ["SDIALOG_LANCEDB_DIR"])

        if lancedb_dir.exists():
            try:
                import lancedb
                db = lancedb.connect(str(lancedb_dir))
                table_name = os.environ.get("SDIALOG_LANCEDB_TABLE", "universities")
                os.environ["SDIALOG_ENABLE_RAG"] = "1" if table_name in db.table_names() else "0"
            except Exception:
                os.environ["SDIALOG_ENABLE_RAG"] = "0"
        else:
            os.environ["SDIALOG_ENABLE_RAG"] = "0"

        return

    embeddings_dir = dataset_dir / "Embeddings"
    faiss_path = embeddings_dir / "rag.index.faiss"
    chunks_path = embeddings_dir / "rag.chunks.jsonl"

    os.environ["SDIALOG_RAG_FAISS"] = str(faiss_path)
    os.environ["SDIALOG_RAG_CHUNKS"] = str(chunks_path)
    os.environ["SDIALOG_ENABLE_RAG"] = "1" if faiss_path.exists() and chunks_path.exists() else "0"


class TimedRetriever:
    def __init__(self, retriever, warn_s: float = 6.0):
        self._r = retriever
        self.warn_s = float(warn_s)

    def __getattr__(self, name: str):
        # Forward optional retriever methods such as keyword_search().
        return getattr(self._r, name)

    def search(self, query: str, top_k: int = 4, **kwargs):
        t0 = time.perf_counter()
        out = self._r.search(query, top_k=top_k, **kwargs)
        dt = time.perf_counter() - t0

        if dt > self.warn_s and os.getenv("SDIALOG_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(f"[TimedRetriever] search dt={dt:.2f}s | query={query!r} | kwargs={kwargs!r}")

        return out

    def keyword_search(self, **kwargs):
        if not hasattr(self._r, "keyword_search"):
            return []
        t0 = time.perf_counter()
        out = self._r.keyword_search(**kwargs)
        dt = time.perf_counter() - t0

        if dt > self.warn_s and os.getenv("SDIALOG_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(f"[TimedRetriever] keyword_search dt={dt:.2f}s | kwargs={kwargs!r}")

        return out


# =============================================================================
# Personas + Agents
# =============================================================================
def create_personas(
        student_json_path: str = "",
        dialog_language: str = "English",
        social_practice_name: str = "university_counseling",
        social_practice_path: str | None = None,
) -> Tuple[Persona, Student]:
    student = load_student_from_json(
        student_json_path,
        dialog_language=dialog_language,
        social_practice_name=social_practice_name,
        social_practice_path=social_practice_path,
    )
    practice = getattr(student, "social_practice", None) or get_social_practice(
        social_practice_name,
        path=social_practice_path,
    )

    counselor = Persona(
        name="University Counselor",
        age="middle-aged",
        gender="unspecified",
        role=practice.get("agent1_role", "UniversityCounselor"),
        background="Works at the university counseling office in Italy.",
        personality="",
        circumstances=f"Social practice: {practice.get('sp_name', 'University Counseling')}",
        rules="; ".join(practice.get("agent1_norms", []) or []),
        language=_lang_name(dialog_language),
    )
    return counselor, student


def create_agents_offline(
        esperto_persona: Persona,
        studente_persona: Student,
        local_model_name_expert: str,
        local_model_name_student: str,
        dialog_language: str = "English",
        social_practice_name: str = "university_counseling",
        social_practice_path: str | None = None,
) -> Tuple[Agent, Agent]:
    practice = getattr(studente_persona, "social_practice", None) or get_social_practice(
        social_practice_name,
        path=social_practice_path,
    )

    set_student_dialog_language(dialog_language)
    set_counselor_dialog_language(dialog_language)

    expert_agent = Agent(
        persona=esperto_persona,
        model=local_model_name_expert,
        name="EXPERT",
        response_details=_build_expert_response_details(practice, dialog_language),
        think=os.getenv("SDIALOG_EXPERT_THINK", "0").lower() in {"1", "true", "yes"},
        postprocess_fn=sanitize_expert_output,
    )

    student_agent = Agent(
        persona=studente_persona,
        model=local_model_name_student,
        name="STUDENT",
        response_details=_build_student_response_details(studente_persona, practice, dialog_language),
        think=False,
        postprocess_fn=sanitize_student_output,
    )

    student_agent = student_agent | StudentFixedFollowupOrchestrator(
        dialog_language=dialog_language,
        practice=practice,
    )

    _configure_rag_env(dialog_language)

    if os.environ.get("SDIALOG_ENABLE_RAG", "0") == "1":
        backend = os.environ.get("SDIALOG_RAG_BACKEND", "lancedb").lower()

        if backend == "lancedb":
            from RAG_Scripts.lancedb_university_retriever import LanceDBUniversityRetriever
            retriever = TimedRetriever(
                LanceDBUniversityRetriever(
                    lang=os.environ.get("SDIALOG_RAG_LANG", "eng"),
                    lancedb_dir=os.environ.get("SDIALOG_LANCEDB_DIR"),
                    table_name=os.environ.get("SDIALOG_LANCEDB_TABLE", "universities"),
                )
            )
        else:
            from RAG_Scripts.rag_retriever import RAGRetriever

            retriever = TimedRetriever(
                RAGRetriever(
                    lang=os.environ.get("SDIALOG_RAG_LANG", "eng"),
                    faiss_path=os.environ.get("SDIALOG_RAG_FAISS", ""),
                    chunks_path=os.environ.get("SDIALOG_RAG_CHUNKS", ""),
                )
            )
        expert_agent = expert_agent | UniversityCounselorFlowOrchestrator(
            retriever=retriever,
            required_slots=("academic_background", "field_of_interest", "region"),
            top_k=12,
            candidate_pool_size=8,
            max_ctx_chars=2200,
            history_turns=3,
            debug=False,
            dialog_language=dialog_language,
            practice=practice,
        )

    return expert_agent, student_agent
