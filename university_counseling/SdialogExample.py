# SdialogExample.py (CLEAN + interleaving, LISTENER_PATCH version)
from __future__ import annotations

import io
import json
import os
import sys
import traceback
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict

from university_counseling.agents_setup import create_agents_offline, create_personas
from university_counseling.AgentWithRag import (
    clear_debug_snapshots,
    get_debug_snapshots,
    clear_listener_patches,
    get_listener_patches,
    apply_listener_patch,
    canonical_listener_memory,
)

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s or "")


os.environ.setdefault("SDIALOG_DEBUG", "1")


class Tee(io.TextIOBase):
    """Duplicate stdout/stderr to console + file, filtering progress bars + stripping ANSI in file."""
    def __init__(self, console_stream: io.TextIOBase, file_stream: io.TextIOBase):
        self.console_stream = console_stream
        self.file_stream = file_stream

    def write(self, s: str) -> int:
        # drop tqdm-ish progress lines
        if "Batches:" in s or "%|" in s or "it/s" in s:
            return len(s)

        s = s.replace("\r", "")

        # Console: keep colors
        self.console_stream.write(s)

        # File: strip colors
        self.file_stream.write(strip_ansi(s))
        return len(s)

    def flush(self) -> None:
        self.console_stream.flush()
        self.file_stream.flush()


def _save_dialog(dialog, out_json: Path) -> None:
    if hasattr(dialog, "to_file"):
        dialog.to_file(str(out_json))
        return
    if hasattr(dialog, "dict"):
        out_json.write_text(json.dumps(dialog.dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    raise RuntimeError("Dialog object does not support to_file() or dict().")





MILESTONE_TURNS = {1, 4, 12, 13, 14}
def _interleave_listener_memory(dialog_text: str) -> str:
    from university_counseling.AgentWithRag import canonical_listener_memory, apply_listener_patch, get_listener_patches

    THINKING_FROM_TURN = 14

    patches = get_listener_patches()
    pi = 0
    mem = canonical_listener_memory()

    out_lines: list[str] = []
    student_turn = 0

    expect_patch = False
    pending_src_turn: int | None = None

    for line in dialog_text.splitlines(True):
        # Track student turns
        if line.startswith("[STUDENT]"):
            student_turn += 1

        # Detect patch request in orchestration
        if line.startswith("[instruct-UniversityCounselorFlowOrchestrator]") and "HIDDEN LISTENER TASK" in line:
            expect_patch = True
            pending_src_turn = student_turn  # patch refers to latest student message

        # Handle expert thinking lines:
        # - keep applying listener patches
        # - print thinking only from STUDENT turn 15 onward
        if line.startswith("[EXPERT]") and "(thinking)" in line:
            src = pending_src_turn or student_turn

            if expect_patch and pi < len(patches):
                mem = apply_listener_patch(mem, patches[pi])
                pi += 1
                if src in MILESTONE_TURNS:
                    out_lines.append(f"[Listener memory | extracted from STUDENT turn {src}]\n")
                    out_lines.append(json.dumps(mem, ensure_ascii=False, indent=2, sort_keys=True) + "\n\n")
                expect_patch = False
                pending_src_turn = None

            if src >= THINKING_FROM_TURN:
                out_lines.append(line)

            continue

        # Print the original line (non-thinking)
        out_lines.append(line)

        # Fallback: if there is no thinking line, apply+print right after the visible EXPERT line
        if line.startswith("[EXPERT]") and expect_patch:
            if pi < len(patches):
                mem = apply_listener_patch(mem, patches[pi])
                pi += 1
                src = pending_src_turn or student_turn
                if src in MILESTONE_TURNS:
                    out_lines.append(f"[Listener memory | extracted from STUDENT turn {src}]\n")
                    out_lines.append(json.dumps(mem, ensure_ascii=False, indent=2, sort_keys=True) + "\n\n")
            expect_patch = False
            pending_src_turn = None

    return "".join(out_lines)

def main() -> None:
    MAX_TURNS = 100
    BASE_SEED = 12345
    DIALOG_LANGUAGE = "English"
    SOCIAL_PRACTICE_NAME = "university_counseling"
    START_PERSONA = 0
    MAX_PERSONAS = 10

    project_root = Path(__file__).resolve().parents[1]

    SOCIAL_PRACTICE_PATH = str(
        project_root / "configuration_data" / "social_practices.json"
    )

    local_model_name_student = "ollama:qwen3:30b"
    local_model_name_expert = "ollama:qwen3:30b-thinking"

    input_dir = project_root / "PERSONAS_DATASET"

    out_base_dir = project_root / "Generated Dialogs"
    out_dialog_dir = out_base_dir / "_out_dialog_json"
    out_text_dir = out_base_dir / "_out_logs_txt"
    out_patches_dir = out_base_dir / "_out_listener_patches"

    out_dialog_dir.mkdir(parents=True, exist_ok=True)
    out_text_dir.mkdir(parents=True, exist_ok=True)
    out_patches_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {input_dir.resolve()}")

    json_files = sorted([p for p in input_dir.glob("*.json") if p.is_file()])
    if not json_files:
        print(f"No .json files found in: {input_dir.resolve()}")
        return
    if MAX_PERSONAS is not None:
        json_files = json_files[START_PERSONA:START_PERSONA + MAX_PERSONAS]
    else:
        json_files = json_files[START_PERSONA:]

    for local_i, json_path in enumerate(json_files):
        global_i = START_PERSONA + local_i

        out_txt = out_text_dir / f"{json_path.stem}.txt"
        out_dialog_json = out_dialog_dir / f"{json_path.stem}.dialog.json"
        out_patches_json = out_patches_dir / f"{json_path.stem}.listener_patches.json"

        print(f"\n\n===== RUNNING: {json_path.name} =====")
        print(f"Log     -> {out_txt.name}")
        print(f"Dialog  -> {out_dialog_json.name}")
        print(f"Patches -> {out_patches_json.name}")

        success = False

        with out_txt.open("w", encoding="utf-8") as f:
            tee_out = Tee(sys.stdout, f)
            tee_err = Tee(sys.stderr, f)

            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                try:
                    # reset buffers for this run
                    clear_debug_snapshots()
                    clear_listener_patches()

                    persona_expert, persona_student = create_personas(
                        str(json_path),
                        dialog_language=DIALOG_LANGUAGE,
                        social_practice_name=SOCIAL_PRACTICE_NAME,
                        social_practice_path=SOCIAL_PRACTICE_PATH,
                    )

                    expert_agent, student_agent = create_agents_offline(
                        persona_expert,
                        persona_student,
                        local_model_name_expert,
                        local_model_name_student,
                        dialog_language=DIALOG_LANGUAGE,
                        social_practice_name=SOCIAL_PRACTICE_NAME,
                        social_practice_path=SOCIAL_PRACTICE_PATH,
                    )

                    seed = BASE_SEED + global_i
                    dialog = student_agent.dialog_with(expert_agent, max_turns=MAX_TURNS, seed=seed)

                    # capture dialog.print output
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        try:
                            dialog.print(all=True, orchestration=True)
                        except Exception:
                            dialog.print()
                    dialog_text = strip_ansi(buf.getvalue())

                    # interleave snapshots (diffs)
                    snapshots = get_debug_snapshots()
                    print(_interleave_listener_memory(dialog_text), end="")

                    # save dialog + patches
                    _save_dialog(dialog, out_dialog_json)
                    out_patches_json.write_text(
                        json.dumps(get_listener_patches(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    success = True

                except Exception:
                    print("\n[ERROR] Exception while processing:", json_path.name, file=sys.stderr)
                    traceback.print_exc()

        if success:
            print(f"[SAVED] {out_txt.resolve()}")
            print(f"[SAVED] {out_dialog_json.resolve()}")
            print(f"[SAVED] {out_patches_json.resolve()}")
        else:
            try:
                if out_dialog_json.exists():
                    out_dialog_json.unlink()
                if out_patches_json.exists():
                    out_patches_json.unlink()
            except Exception:
                pass
            print(f"[FAILED] {json_path.name} (see log: {out_txt.resolve()})")

    print("\nDONE.")


if __name__ == "__main__":
    main()
