from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from sdialog.agents import Agent
from sdialog.personas import Persona

from university_counseling.AgentWithRag import (
    UniversityCounselorFlowOrchestrator,
    apply_listener_patch,
    canonical_listener_memory,
    clear_debug_snapshots,
    clear_last_rag_state,
    clear_listener_patches,
    get_debug_snapshots,
    get_listener_patch_events,
    get_listener_patches,
    get_selected_option_event,
    set_dialog_language as set_rag_dialog_language,
)
from university_counseling.DialogOrchestrator import (
    StudentFixedFollowupOrchestrator,
    clear_student_script_state,
    enforce_student_script_or_fallback,
    set_dialog_language as set_student_dialog_language,
)
from university_counseling.agents_setup import (
    TimedRetriever,
    _build_expert_response_details,
    _configure_rag_env,
    sanitize_expert_output,
    strip_think,
)
from university_counseling.social_practice import (
    get_practice_list,
    get_social_practice,
    student_prompt_addendum,
)
from university_counseling.student_profile import load_student_from_json

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


class Tee(io.TextIOBase):
    """Duplicate stdout/stderr to console and file, with quieter file logs."""

    def __init__(self, console_stream: io.TextIOBase, file_stream: io.TextIOBase) -> None:
        self.console_stream = console_stream
        self.file_stream = file_stream

    def write(self, text: str) -> int:
        original_len = len(text or "")
        if not text:
            return 0

        if "Batches:" in text or "%|" in text or "it/s" in text:
            return original_len

        show_orchestration = _env_flag("SDIALOG_LOG_ORCHESTRATION", "0")
        show_thinking = _env_flag("SDIALOG_LOG_THINKING", "0")

        kept: list[str] = []
        for line in text.replace("\r", "").splitlines(True):
            plain = strip_ansi(line).lstrip()
            if not show_orchestration and plain.startswith("[instruct-"):
                continue
            if not show_thinking and plain.startswith("[EXPERT]") and "(thinking)" in plain:
                continue
            kept.append(line)

        cleaned = "".join(kept)
        if not cleaned:
            return original_len

        self.console_stream.write(cleaned)
        self.file_stream.write(strip_ansi(cleaned))
        return original_len

    def flush(self) -> None:
        self.console_stream.flush()
        self.file_stream.flush()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    if raw.lower() in {"all", "none", "*"}:
        return None
    return int(raw)


def _split_env_list(raw: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;,]", raw or "") if x.strip()]


def _normalize_condition_label(value: str, default: str = "condition") -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or default).strip())
    return label.strip("_") or default


def _parse_model_conditions(raw: str) -> list[tuple[str, str]]:
    conditions: list[tuple[str, str]] = []
    for item in _split_env_list(raw):
        if "=" in item:
            label, model = item.split("=", 1)
            label = _normalize_condition_label(label, "model")
            model = model.strip()
        else:
            model = item.strip()
            label = _normalize_condition_label(model.replace("ollama:", ""), "model")
        if label and model:
            conditions.append((label, model))
    return conditions


def _normalize_gender_condition(value: str) -> str:
    raw = (value or "").strip().lower()
    aliases = {
        "m": "male",
        "man": "male",
        "male": "male",
        "maschio": "male",
        "uomo": "male",
        "f": "female",
        "woman": "female",
        "female": "female",
        "femmina": "female",
        "donna": "female",
    }
    if raw not in aliases:
        raise ValueError(f"Invalid gender condition {value!r}. Use male/female.")
    return aliases[raw]


def _lang_name(dialog_language: str = "English") -> str:
    return "Italian" if (dialog_language or "").strip().lower().startswith("it") else "English"


def _rag_lang(dialog_language: str = "English") -> str:
    env_lang = os.getenv("SDIALOG_RAG_LANG", "").strip().lower()
    if env_lang in {"ita", "it", "italian", "italiano"}:
        return "ita"
    if env_lang in {"eng", "en", "english", "inglese"}:
        return "eng"
    return "ita" if _lang_name(dialog_language) == "Italian" else "eng"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(str(x) for x in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _save_dialog(dialog: Any, out_json: Path) -> None:
    if hasattr(dialog, "to_file"):
        dialog.to_file(str(out_json))
        return
    if hasattr(dialog, "dict"):
        out_json.write_text(json.dumps(dialog.dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    out_json.write_text(json.dumps(_json_safe(dialog), ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_input_dir(project_root: Path) -> Path:
    raw = os.getenv("SDIALOG_INPUT_DIR", "PERSONAS_DATASET_baseline_02_riasec_l2_only").strip()
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _load_persona_files(input_dir: Path) -> list[Path]:
    list_file = os.getenv("SDIALOG_PERSONA_LIST", "").strip()
    if not list_file:
        return sorted([p for p in input_dir.glob("*.json") if p.is_file()])

    list_path = Path(list_file)
    if not list_path.is_absolute():
        list_path = input_dir.parent / list_path

    files: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        path = Path(item)
        if not path.is_absolute():
            path = input_dir / path.name
        files.append(path)
    return files


def _restore_env(name: str, old_value: str | None) -> None:
    if old_value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old_value


def _fixed_followups_for_validation(
    practice: dict[str, Any],
    dialog_language: str,
) -> list[str]:
    return get_practice_list(
        practice,
        "dialogue_config",
        "student",
        "fixed_followup_questions",
        dialog_language=dialog_language,
        default=[],
    )


def _dialog_protocol_status(
    dialog_text: str,
    practice: dict[str, Any],
    dialog_language: str,
) -> tuple[bool, str]:
    text = dialog_text or ""
    fixed = _fixed_followups_for_validation(practice, dialog_language)
    missing = [q for q in fixed if q and q not in text]
    if missing:
        return False, f"missing_fixed_followup_questions={len(missing)}"

    if dialog_language.lower().startswith("it"):
        thank_ok = "grazie" in text.lower() and ("arrivederci" in text.lower() or "ciao" in text.lower())
    else:
        thank_ok = "thank you" in text.lower() and "goodbye" in text.lower()
    if not thank_ok:
        return False, "missing_student_thank_you_goodbye"

    recap_rx = re.compile(
    r"(?im)^\[(?:EXPERT|Counselor[^\n]*)\]\s*"
    r"(?:Recap|Riepilogo)\s*:")
    if not recap_rx.search(text):
        return False, "missing_counselor_wrapup_recap"

    tail = text.rstrip().splitlines()[-3:]
    if any(line.startswith("[STUDENT]") and "?" in line for line in tail):
        return False, "ended_after_unanswered_student_question"

    return True, ""


def _build_retriever(dialog_language: str) -> TimedRetriever:
    _configure_rag_env(dialog_language)
    from RAG_Scripts.lancedb_university_retriever import LanceDBUniversityRetriever

    retriever = LanceDBUniversityRetriever(
        lang=_rag_lang(dialog_language),
        lancedb_dir=os.getenv("SDIALOG_LANCEDB_DIR", "").strip() or None,
        table_name=os.getenv("SDIALOG_LANCEDB_TABLE", "universities"),
        embed_model=os.getenv("SDIALOG_EMBED_MODEL", "intfloat/multilingual-e5-base"),
    )
    return TimedRetriever(retriever)


def _build_counselor_agent(
    *,
    model: str,
    dialog_language: str,
    practice: dict[str, Any],
    retriever: TimedRetriever,
    model_label: str,
) -> Agent:
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

    counselor = Agent(
        persona=counselor_persona,
        model=model,
        name=f"Counselor[{model_label}]",
        response_details=_build_expert_response_details(practice, dialog_language),
        think=_env_flag("SDIALOG_EXPERT_THINK", "0"),
        postprocess_fn=sanitize_expert_output,
    )

    return counselor | UniversityCounselorFlowOrchestrator(
        retriever=retriever,
        required_slots=("academic_background", "field_of_interest", "region"),
        top_k=_env_int("SDIALOG_COUNSELOR_TOP_K", 6) or 6,
        candidate_pool_size=_env_int("SDIALOG_CANDIDATE_POOL_SIZE", 4) or 4,
        min_options_target=_env_int("SDIALOG_MIN_OPTIONS_TARGET", 3) or 3,
        final_top_n=_env_int("SDIALOG_FINAL_TOP_N", 3) or 3,
        max_ctx_chars=_env_int("SDIALOG_MAX_CTX_CHARS", 4000) or 4000,
        history_turns=_env_int("SDIALOG_HISTORY_TURNS", 2) or 2,
        debug=_env_flag("SDIALOG_DEBUG", "0"),
        dialog_language=dialog_language,
        practice=practice,
    )


def _build_student_response_details(practice: dict[str, Any], dialog_language: str) -> str:
    lang = _lang_name(dialog_language)
    addendum = student_prompt_addendum(practice, dialog_language)
    return "\n".join(
        [
            f"Always answer in {lang}.",
            "You are the student in a counseling experiment.",
            "The STUDENT SCRIPT instruction for the current turn has priority.",
            "Treat BACKGROUND as authoritative except where it conflicts with the ACADEMIC STATUS rules below.",
            "ACADEMIC STATUS (STRICT): your highest completed formal education is either a high-school diploma or a bachelor's degree, whichever is explicitly present in BACKGROUND.",
            "You are not employed, not a worker or researcher, and you do not have a master's degree, PhD, doctorate, or other postgraduate qualification.",
            "Never describe a job, laboratory role, internship, research activity, or professional title as your academic background.",
            "Treat INTERESTS as preference evidence and PERSONALITY TRAITS only as speaking-style cues.",
            "Do not convert interests or personality traits into new education, work, skills, experiences, or other biographical facts.",
            "Do not reveal hidden experimental conditions or metadata.",
            addendum,
        ]
    ).strip()


def _build_student_agent(
    *,
    persona_path: Path,
    model: str,
    dialog_language: str,
    practice: dict[str, Any],
    social_practice_name: str,
    social_practice_path: str,
) -> Agent:
    student = load_student_from_json(
        str(persona_path),
        dialog_language=dialog_language,
        social_practice_name=social_practice_name,
        social_practice_path=social_practice_path,
    )

    def _student_postprocess(text: str) -> str:
        return enforce_student_script_or_fallback(strip_think(text))

    student_agent = Agent(
        persona=student,
        model=model,
        name="Student",
        response_details=_build_student_response_details(practice, dialog_language),
        think=_env_flag("SDIALOG_STUDENT_THINK", "0"),
        postprocess_fn=_student_postprocess,
    )

    return student_agent | StudentFixedFollowupOrchestrator(
        dialog_language=dialog_language,
        practice=practice,
    )


def _render_dialog(dialog: Any) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            dialog.print(all=True, orchestration=True)
        except Exception:
            dialog.print()
    return strip_ansi(buf.getvalue())


def _final_listener_memory() -> dict[str, Any]:
    memory = canonical_listener_memory()
    for patch in get_listener_patches():
        memory = apply_listener_patch(memory, patch)
    return memory


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _print_run_diagnostics(dialog_text: str) -> None:
    print(dialog_text, end="" if dialog_text.endswith("\n") else "\n")

    patches = get_listener_patches()
    events = get_listener_patch_events()
    selected = get_selected_option_event()
    final_memory = _final_listener_memory()

    print("\n[Listener patches | raw accepted patches]")
    print(json.dumps(patches, ensure_ascii=False, indent=2, sort_keys=True))

    print("\n[Listener events | turn and phase]")
    print(json.dumps(events, ensure_ascii=False, indent=2, sort_keys=True))

    print("\n[Final listener memory | cumulative]")
    print(json.dumps(
        {
            "patch_count": len(patches),
            "memory": final_memory,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))

    print("\n[Selected option | deterministic protocol state]")
    print(json.dumps(selected or None, ensure_ascii=False, indent=2, sort_keys=True))

    if _env_flag("SDIALOG_LOG_DEBUG_SNAPSHOTS", "0"):
        print("\n[Debug snapshots]")
        print(json.dumps(get_debug_snapshots(), ensure_ascii=False, indent=2, sort_keys=True))


def _run_one_dialog(
    *,
    persona_path: Path,
    expert_model: str,
    student_model: str,
    model_label: str,
    gender: str,
    seed: int,
    max_turns: int,
    dialog_language: str,
    practice: dict[str, Any],
    social_practice_name: str,
    social_practice_path: str,
    retriever: TimedRetriever,
) -> tuple[Any, str]:
    set_rag_dialog_language(dialog_language)
    set_student_dialog_language(dialog_language)
    clear_debug_snapshots()
    clear_listener_patches()
    clear_last_rag_state()
    clear_student_script_state()

    counselor_agent = _build_counselor_agent(
        model=expert_model,
        dialog_language=dialog_language,
        practice=practice,
        retriever=retriever,
        model_label=model_label,
    )
    student_agent = _build_student_agent(
        persona_path=persona_path,
        model=student_model,
        dialog_language=dialog_language,
        practice=practice,
        social_practice_name=social_practice_name,
        social_practice_path=social_practice_path,
    )

    print(f"[RUN] persona={persona_path.name} gender_condition={gender} model={model_label}")
    print(f"[MODEL] counselor={expert_model}")
    print(f"[MODEL] student={student_model}")
    print(f"[SEED] {seed}")
    print(f"[LANG] dialog={dialog_language} rag={_rag_lang(dialog_language)}")

    dialog = student_agent.dialog_with(counselor_agent, max_turns=max_turns, seed=seed)
    dialog_text = _render_dialog(dialog)
    _print_run_diagnostics(dialog_text)
    return dialog, dialog_text


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dialog_language = os.getenv("SDIALOG_DIALOG_LANGUAGE", "Italian")
    social_practice_name = os.getenv("SDIALOG_SOCIAL_PRACTICE_NAME", "university_counseling")
    social_practice_path = os.getenv(
        "SDIALOG_SOCIAL_PRACTICE_PATH",
        str(project_root / "configuration_data" / "social_practices.json"),
    )

    max_turns = _env_int("SDIALOG_MAX_TURNS", 100) or 100
    base_seed = _env_int("SDIALOG_BASE_SEED", 12345) or 12345
    start_persona = _env_int("SDIALOG_START_PERSONA", 0) or 0
    max_personas = _env_int("SDIALOG_MAX_PERSONAS", None)
    experiment_name = os.getenv("SDIALOG_EXPERIMENT_NAME", "baseline_gender_ollama").strip()

    model_conditions = _parse_model_conditions(
        os.getenv("SDIALOG_MODEL_CONDITIONS", "gemma_4_31b=ollama:gemma-4-31B-it")
    )
    if not model_conditions:
        raise ValueError("No model conditions found. Set SDIALOG_MODEL_CONDITIONS='label=ollama:model'.")

    fixed_student_model = os.getenv("SDIALOG_STUDENT_MODEL", "").strip()
    same_student_as_expert = _env_flag("SDIALOG_STUDENT_MODEL_SAME_AS_EXPERT", "1")

    gender_variants = [_normalize_gender_condition(x) for x in _split_env_list(os.getenv("SDIALOG_GENDER_VARIANTS", "male,female"))]
    if not gender_variants:
        raise ValueError("No gender variants found. Set SDIALOG_GENDER_VARIANTS='male,female'.")

    input_dir = _resolve_input_dir(project_root)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir.resolve()}")

    persona_files = _load_persona_files(input_dir)
    missing = [str(p) for p in persona_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Some files from SDIALOG_PERSONA_LIST were not found:\n" + "\n".join(missing[:20]))
    if not persona_files:
        print(f"No .json files found in: {input_dir.resolve()}")
        return
    persona_files = persona_files[start_persona:start_persona + max_personas] if max_personas is not None else persona_files[start_persona:]

    practice = get_social_practice(social_practice_name, path=social_practice_path)
    retriever = _build_retriever(dialog_language)

    out_base_dir = project_root / "Generated Dialogs" / experiment_name
    out_base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_base_dir / "run_manifest.csv"

    old_env = {
        "SDIALOG_FORCED_GENDER": os.environ.get("SDIALOG_FORCED_GENDER"),
        "OWUI_FORCED_GENDER": os.environ.get("OWUI_FORCED_GENDER"),
        "SDIALOG_PERSONA_GENDER_CONDITION": os.environ.get("SDIALOG_PERSONA_GENDER_CONDITION"),
        "SDIALOG_STUDENT_GENDER_CONDITION": os.environ.get("SDIALOG_STUDENT_GENDER_CONDITION"),
        "SDIALOG_HIDE_VISIBLE_GENDER_CUES": os.environ.get("SDIALOG_HIDE_VISIBLE_GENDER_CUES"),
        "SDIALOG_EXPOSE_SENSITIVE_CONTEXT": os.environ.get("SDIALOG_EXPOSE_SENSITIVE_CONTEXT"),
        "SDIALOG_ALLOW_FORCED_LISTENER_GENDER": os.environ.get("SDIALOG_ALLOW_FORCED_LISTENER_GENDER"),
        "SDIALOG_USE_GENDERED_NAMES": os.environ.get("SDIALOG_USE_GENDERED_NAMES"),
    }

    rows: list[dict[str, str]] = []

    try:
        os.environ["SDIALOG_EXPOSE_SENSITIVE_CONTEXT"] = "0"
        os.environ["SDIALOG_ALLOW_FORCED_LISTENER_GENDER"] = "1"
        os.environ.setdefault("SDIALOG_HIDE_VISIBLE_GENDER_CUES", "1")
        os.environ.setdefault("SDIALOG_USE_GENDERED_NAMES", "0")
        os.environ.pop("SDIALOG_PERSONA_GENDER_CONDITION", None)
        os.environ.pop("SDIALOG_STUDENT_GENDER_CONDITION", None)

        for model_label, expert_model in model_conditions:
            student_model = expert_model if same_student_as_expert or not fixed_student_model else fixed_student_model

            for gender in gender_variants:
                gender_label = _normalize_condition_label(gender, "gender")
                os.environ["SDIALOG_FORCED_GENDER"] = gender
                os.environ["OWUI_FORCED_GENDER"] = gender
                os.environ.pop("SDIALOG_PERSONA_GENDER_CONDITION", None)
                os.environ.pop("SDIALOG_STUDENT_GENDER_CONDITION", None)

                out_dir = out_base_dir / model_label / f"gender_{gender_label}"
                out_dialog_dir = out_dir / "_out_dialog_json"
                out_text_dir = out_dir / "_out_logs_txt"
                out_patches_dir = out_dir / "_out_listener_patches"
                out_events_dir = out_dir / "_out_listener_events"
                out_memory_dir = out_dir / "_out_listener_memory"
                out_selected_dir = out_dir / "_out_selected_options"
                for path in (out_dialog_dir, out_text_dir, out_patches_dir, out_events_dir, out_memory_dir, out_selected_dir):
                    path.mkdir(parents=True, exist_ok=True)

                for local_i, persona_path in enumerate(persona_files):
                    global_i = start_persona + local_i
                    seed = base_seed + global_i
                    run_id = f"{persona_path.stem}__gender_{gender_label}__model_{model_label}"

                    out_txt = out_text_dir / f"{run_id}.txt"
                    out_dialog_json = out_dialog_dir / f"{run_id}.dialog.json"
                    out_patches_json = out_patches_dir / f"{run_id}.listener_patches.json"
                    out_events_json = out_events_dir / f"{run_id}.listener_events.json"
                    out_memory_json = out_memory_dir / f"{run_id}.listener_memory.json"
                    out_selected_json = out_selected_dir / f"{run_id}.selected_option.json"

                    success = False
                    protocol_ok = False
                    protocol_error = ""
                    error = ""
                    selected_event: dict[str, Any] = {}

                    with out_txt.open("w", encoding="utf-8") as log_file:
                        tee_out = Tee(sys.stdout, log_file)
                        tee_err = Tee(sys.stderr, log_file)
                        with redirect_stdout(tee_out), redirect_stderr(tee_err):
                            try:
                                dialog, dialog_text = _run_one_dialog(
                                    persona_path=persona_path,
                                    expert_model=expert_model,
                                    student_model=student_model,
                                    model_label=model_label,
                                    gender=gender,
                                    seed=seed,
                                    max_turns=max_turns,
                                    dialog_language=dialog_language,
                                    practice=practice,
                                    social_practice_name=social_practice_name,
                                    social_practice_path=social_practice_path,
                                    retriever=retriever,
                                )

                                _write_json(out_patches_json, get_listener_patches())
                                _write_json(out_events_json, get_listener_patch_events())
                                _write_json(out_memory_json, _final_listener_memory())
                                selected_event = get_selected_option_event() or {}
                                _write_json(out_selected_json, selected_event or None)

                                protocol_ok, protocol_error = _dialog_protocol_status(dialog_text, practice, dialog_language)
                                try:
                                    dialog.complete = bool(protocol_ok)
                                except Exception:
                                    pass

                                _save_dialog(dialog, out_dialog_json)

                                if not protocol_ok:
                                    print(f"[INVALID_PROTOCOL] {protocol_error}")
                                success = True
                            except Exception as exc:
                                error = repr(exc)
                                print(f"\n[ERROR] Exception while processing {persona_path.name}", file=sys.stderr)
                                traceback.print_exc()

                    status = "success" if success and protocol_ok else ("invalid_protocol" if success else "failed")
                    if status == "failed":
                        print(f"[FAILED] {persona_path.name} (see log: {out_txt.resolve()})")
                    elif status == "invalid_protocol":
                        print(f"[SAVED INVALID_PROTOCOL] {out_txt.resolve()}")
                    else:
                        print(f"[SAVED] {out_txt.resolve()}")

                    rows.append(
                        {
                            "experiment": experiment_name,
                            "run_id": run_id,
                            "persona_file": persona_path.name,
                            "persona_path": str(persona_path),
                            "gender_condition": gender_label,
                            "model_label": model_label,
                            "expert_model": expert_model,
                            "student_model": student_model,
                            "seed": str(seed),
                            "status": status,
                            "error": error or protocol_error,
                            "protocol_ok": str(bool(protocol_ok)),
                            "protocol_error": protocol_error,
                            "log_path": str(out_txt),
                            "dialog_json_path": str(out_dialog_json),
                            "patches_json_path": str(out_patches_json),
                            "events_json_path": str(out_events_json),
                            "memory_json_path": str(out_memory_json),
                            "selected_options_path": str(out_selected_json),
                            "selected_option_number": str(selected_event.get("option_number", "")) if isinstance(selected_event, dict) else "",
                            "selected_option": str(selected_event.get("selected_option", "")) if isinstance(selected_event, dict) else "",
                            "selected_option_source_turn": str(selected_event.get("source_turn", "")) if isinstance(selected_event, dict) else "",
                            "gender_policy": "hidden_counselor_listener_gender_no_visible_student_gender_cue_by_default",
                            "riasec_policy": "inferred_by_counselor_listener_no_source_riasec_injection",
                        }
                    )

                    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
                        writer = csv.DictWriter(manifest_file, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
    finally:
        for name, old_value in old_env.items():
            _restore_env(name, old_value)

    print(f"\nDONE. Manifest -> {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
