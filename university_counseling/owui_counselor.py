# university_counseling/owui_counselor.py

from __future__ import annotations

import os
import re

from pathlib import Path

from sdialog.agents import Agent
from sdialog.personas import Persona

from university_counseling.agents_setup import (
    TimedRetriever,
    _build_expert_response_details,
    _configure_rag_env,
    sanitize_expert_output,
)
from university_counseling.social_practice import get_social_practice
from university_counseling.AgentWithRag import (
    UniversityCounselorFlowOrchestrator,
    clear_debug_snapshots,
    clear_listener_patches,
    set_dialog_language as set_counselor_dialog_language,
)

import json
from datetime import datetime
from pathlib import Path

from university_counseling.AgentWithRag import (
    get_listener_patches,
    apply_listener_patch,
    canonical_listener_memory,
)

_LISTENER_MEM = canonical_listener_memory()
_PATCH_COUNT = 0

_LAST_PRINTED_MEM = None


def reset_listener_log_state() -> None:
    global _LISTENER_MEM, _PATCH_COUNT, _LAST_PRINTED_MEM

    _LISTENER_MEM = canonical_listener_memory()
    _PATCH_COUNT = 0
    _LAST_PRINTED_MEM = None

    clear_listener_patches()
    clear_debug_snapshots()

_LAST_PRINTED_MEM = None


def sanitize_and_log_expert_output(text: str) -> str:
    global _LISTENER_MEM, _PATCH_COUNT, _LAST_PRINTED_MEM

    visible = sanitize_expert_output(text)

    patches = get_listener_patches()
    new_patches = patches[_PATCH_COUNT:]

    if not new_patches:
        return visible

    for patch in new_patches:
        _LISTENER_MEM = apply_listener_patch(_LISTENER_MEM, patch)

    _PATCH_COUNT = len(patches)

    mem_json = json.dumps(
        _LISTENER_MEM,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    if mem_json != _LAST_PRINTED_MEM:
        print("\n[LISTENER MEMORY]", flush=True)
        print(mem_json, flush=True)
        _LAST_PRINTED_MEM = mem_json

    return visible

def _lang_name(dialog_language: str) -> str:
    return "Italian" if dialog_language.lower().startswith("it") else "English"

_RESET_COMMAND_RX = re.compile(
    r"^\s*(/reset|/new|nuova chat|nuova conversazione|ricominciamo|ripartiamo)\s*$",
    re.IGNORECASE,
)


class ResetOnNewChatCounselorFlowOrchestrator(UniversityCounselorFlowOrchestrator):
    def instruct(self, dialog, utterance):
        student_utt = (utterance or "").strip()

        if _RESET_COMMAND_RX.match(student_utt):
            self.reset()
            reset_listener_log_state()
            print("\n[RESET] Counselor state and listener memory reset.", flush=True)

            # Dopo il reset, facciamo partire il counselor come se fosse una nuova conversazione.
            return super().instruct(dialog, "")

        return super().instruct(dialog, utterance)


def build_owui_counselor() -> Agent:
    project_root = Path(__file__).resolve().parents[1]

    dialog_language = os.getenv("DIALOG_LANGUAGE", "Italian")
    social_practice_name = os.getenv("SOCIAL_PRACTICE_NAME", "university_counseling")
    social_practice_path = os.getenv(
        "SOCIAL_PRACTICE_PATH",
        str(project_root / "configuration_data" / "social_practices.json"),
    )

    local_model_name_expert = os.getenv(
        "COUNSELOR_MODEL",
        "ollama:qwen3:30b-thinking",
    )

    practice = get_social_practice(
        social_practice_name,
        path=social_practice_path,
    )

    set_counselor_dialog_language(dialog_language)

    counselor_persona = Persona(
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

    counselor_agent = Agent(
        persona=counselor_persona,
        model=local_model_name_expert,
        name="university-counselor",
        response_details=_build_expert_response_details(practice, dialog_language),
        think=True,
        postprocess_fn=sanitize_and_log_expert_output,
    )

    _configure_rag_env()

    if os.environ.get("SDIALOG_ENABLE_RAG", "0") != "1":
        backend = os.environ.get("SDIALOG_RAG_BACKEND", "lancedb").lower()

        if backend == "lancedb":
            raise RuntimeError(
                "RAG non abilitato: non trovo l'indice LanceDB. "
                "Esegui prima: python RAG_Scripts/Build_lancedb_index_university.py "
                "e controlla il path RAG_University/Embeddings/lancedb."
            )

        raise RuntimeError(
            "RAG non abilitato: non trovo rag.index.faiss e rag.chunks.jsonl. "
            "Ricostruisci l'indice FAISS o controlla i path in RAG_University/Embeddings."
        )

    backend = os.environ.get("SDIALOG_RAG_BACKEND", "lancedb").lower()

    if backend == "lancedb":
        from RAG_Scripts.lancedb_university_retriever import LanceDBUniversityRetriever  # type: ignore

        retriever = TimedRetriever(
            LanceDBUniversityRetriever(
                lancedb_dir=os.environ.get(
                    "SDIALOG_LANCEDB_DIR",
                    str(project_root / "RAG_University" / "Embeddings" / "lancedb"),
                ),
                table_name=os.environ.get("SDIALOG_LANCEDB_TABLE", "universities"),
            )
        )
    else:
        from RAG_Scripts.rag_retriever import RAGRetriever  # type: ignore

        retriever = TimedRetriever(
            RAGRetriever(
                faiss_path=os.environ.get("SDIALOG_RAG_FAISS", ""),
                chunks_path=os.environ.get("SDIALOG_RAG_CHUNKS", ""),
            )
        )

    counselor_agent = counselor_agent | ResetOnNewChatCounselorFlowOrchestrator(
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

    clear_listener_patches()
    clear_debug_snapshots()

    return counselor_agent


def main() -> None:
    agent = build_owui_counselor()
    agent.serve(port=int(os.getenv("OWUI_AGENT_PORT", "1333")))


if __name__ == "__main__":
    main()