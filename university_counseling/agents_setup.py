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
        "If you are given a 'HIDDEN LISTENER TASK (STRICT)', append exactly ONE <LISTENER_PATCH>...</LISTENER_PATCH> JSON block after your raw answer. Do not mention it in the visible reply.\n"
        "Inside <LISTENER_PATCH>, keep JSON keys in English exactly as requested.\n"
        "Do not guess sensitive attributes.\n"
    ).strip()

def _build_student_response_details(student: Student, practice: dict, dialog_language: str = "English") -> str:
    lang = _lang_name(dialog_language)
    rules = (getattr(student, "rules", "") or "").strip()
    background = (getattr(student, "background", "") or "").strip()

    return (
        f"Always answer in {lang}. "
        "Do not invent facts about yourself. "
        "If a detail is not in your BACKGROUND, say you don't know or haven't decided. "
        "If you receive an orchestration instruction, follow it EXACTLY. "
        "Do not pretend that an option exists unless the counselor has presented grounded numbered options. "
        f"{student_prompt_addendum(practice, dialog_language)}\n\n"
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
    return cleaned


def sanitize_expert_output(text: str) -> str:
    # 1) Remove model thinking first.
    # This prevents accidental listener patches inside <think> from being accepted.
    t = strip_think(text)

    # 2) Extract and buffer listener patches only from the final visible channel.
    t = strip_and_buffer_listener_patch(t)

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

    return t


def sanitize_student_output(text: str) -> str:
    t = strip_think(text)
    t = enforce_student_script_or_fallback(t)
    return t


# =============================================================================
# RAG wiring
# =============================================================================
def _configure_rag_env() -> None:
    project_root = Path(__file__).resolve().parents[1]

    candidate_dirs = [
        project_root / "RAG_University" / "Embeddings",
        project_root / "RAG_University",
        project_root / "RAG_Scripts",
    ]

    for rag_dir in candidate_dirs:
        faiss_path = rag_dir / "rag.index.faiss"
        chunks_path = rag_dir / "rag.chunks.jsonl"

        if faiss_path.exists() and chunks_path.exists():
            os.environ["SDIALOG_RAG_FAISS"] = str(faiss_path)
            os.environ["SDIALOG_RAG_CHUNKS"] = str(chunks_path)
            os.environ["SDIALOG_ENABLE_RAG"] = "1"
            return

    fallback_dir = project_root / "RAG_University" / "Embeddings"
    os.environ["SDIALOG_RAG_FAISS"] = str(fallback_dir / "rag.index.faiss")
    os.environ["SDIALOG_RAG_CHUNKS"] = str(fallback_dir / "rag.chunks.jsonl")
    os.environ["SDIALOG_ENABLE_RAG"] = "0"


class TimedRetriever:
    def __init__(self, retriever, warn_s: float = 6.0):
        self._r = retriever
        self.warn_s = float(warn_s)

    def search(self, query: str, top_k: int = 4, **kwargs):
        t0 = time.perf_counter()
        out = self._r.search(query, top_k=top_k, **kwargs)
        dt = time.perf_counter() - t0

        if dt > self.warn_s and os.getenv("SDIALOG_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(f"[TimedRetriever] search dt={dt:.2f}s | query={query!r} | kwargs={kwargs!r}")

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
        think=True,
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

    _configure_rag_env()

    if os.environ.get("SDIALOG_ENABLE_RAG", "0") == "1":
        from RAG_Scripts.rag_retriever import RAGRetriever  # type: ignore

        retriever = TimedRetriever(
            RAGRetriever(
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
