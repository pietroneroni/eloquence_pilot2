from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, List

from sdialog.orchestrators import BaseOrchestrator

from .social_practice import (
    default_university_counseling_practice,
    get_practice_list,
)
from .AgentWithRag import get_last_rag_phase

def _lang_code(dialog_language: str = "English") -> str:
    return "it" if (dialog_language or "").strip().lower().startswith("it") else "en"

_DIALOG_LANGUAGE = "en"

def set_dialog_language(dialog_language: str = "English") -> None:
    global _DIALOG_LANGUAGE
    _DIALOG_LANGUAGE = _lang_code(dialog_language)

def _t(en: str, it: str) -> str:
    return it if _DIALOG_LANGUAGE == "it" else en

def _safe_turn_text(turn) -> str:
    if turn is None:
        return ""
    if isinstance(turn, dict):
        for k in ("utterance", "text", "content", "message"):
            v = turn.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    for attr in ("utterance", "text", "content", "message"):
        if hasattr(turn, attr):
            v = getattr(turn, attr)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


_OPT_RX = re.compile(
    r"\b(?:option|opzione)\s*([1-3])\b|\b([1-3])\b",
    re.I,
)



@dataclass
class _StudentScriptState:
    mode: str = ""
    expected_question: str = ""


_LAST_STUDENT_SCRIPT_STATE = _StudentScriptState()


def set_student_script_state(mode: str, expected_question: str = "") -> None:
    _LAST_STUDENT_SCRIPT_STATE.mode = (mode or "").strip().lower()
    _LAST_STUDENT_SCRIPT_STATE.expected_question = (expected_question or "").strip()


def clear_student_script_state() -> None:
    set_student_script_state("", "")


def _strip_questions_from_text(text: str, max_sentences: int = 2) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return "I understand."

    raw_parts = re.split(r"(?<=[.!?])\s+", t)
    kept: List[str] = []

    for part in raw_parts:
        p = part.strip()
        if not p:
            continue
        if "?" in p:
            continue
        kept.append(p)
        if len(kept) >= max_sentences:
            break

    if not kept:
        # fallback ultra-safe
        cleaned = t.replace("?", "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            return "I understand."
        if cleaned[-1] not in ".!":
            cleaned += "."
        return cleaned

    out = " ".join(kept).strip()
    if out[-1] not in ".!":
        out += "."
    return out


def enforce_student_script_or_fallback(text: str) -> str:
    t = (text or "").strip()
    mode = (_LAST_STUDENT_SCRIPT_STATE.mode or "").strip().lower()
    expected = (_LAST_STUDENT_SCRIPT_STATE.expected_question or "").strip()

    if not mode:
        return t

    if mode == "goodbye_only":
        return _t("Goodbye", "Arrivederci")

    if mode == "thank_goodbye":
        return  _t(
            "Thank you for your help, goodbye.",
            "Grazie per il tuo aiuto, arrivederci."
        )

    if mode == "answer_only_no_question":
        return _strip_questions_from_text(t, max_sentences=2)

    if not expected:
        return t

    if mode == "followup_only":
        return expected

    if mode == "choose_and_q1":
        m = _OPT_RX.search(t)
        if not m:
            raise ValueError(
                "Student output did not contain a numbered option; "
                "refusing silent option-1 fallback."
            )
        n = m.group(1) or m.group(2)
        choice_line = _t(
            f"I will choose option {n}.",
            f"Scelgo l'opzione {n}."
        )
        return f"{choice_line}\n{expected}".strip()

    return t

class StudentFixedFollowupOrchestrator(BaseOrchestrator):
    QUESTIONS: List[str] = [
        "Do you think I could handle a demanding technical course?",
        "Would you recommend focusing on a technical program or a program oriented toward humanities?",
        "Why do you think that choice would suit me?",
        "Do you think I would feel comfortable leading a group project?",
        "Would you suggest I take a leadership role now or focus on supporting tasks first?",
        "Why do you think one option is more suitable for me?",
        "Do you think my previous studies have adequately prepared me for university-level courses?",
        "Would you suggest I take advanced classes immediately or start with introductory modules?",
        "Why do you think that approach fits my preparation?",
    ]

    BACKGROUND_QUESTIONS: List[str] = [
        "What is your current education level (e.g., high school type, bachelor) and what subjects have you studied most recently?",
        "What subjects/topics do you enjoy most, and what are your strengths vs weaknesses? (Grades if you want to share.)",
        "What are your goals after graduation, and what constraints matter most (budget, language, mobility, preferred city/region in Italy)?",
    ]

    RIASEC_QUESTIONS: List[str] = [
        "Do you like practical, hands-on work—building or repairing things?",
        "Are you curious to understand how things work?",
        "Do you enjoy researching, analyzing problems, and reasoning logically?",
        "Do you like expressing what you think or feel through creative forms (writing, art, music, design)?",
        "Do you like experimenting and creating new things?",
        "Do you like working closely with people and having a supportive/helping role?",
        "Do you like proposing ideas and organizing projects?",
        "Do you like having everything organized and under control?",
    ]

    _OPTIONS_RX = re.compile(
        r"(?:\bwhich option do you want to pursue\b|^\s*(?:Option\s*1\b|1[\.\)]\s+)|\bChoose\s+1\b)",
        re.IGNORECASE | re.MULTILINE,
    )

    _GOODBYE_RX = re.compile(r"\bgoodbye\b", re.IGNORECASE)

    def __init__(
            self,
            dialog_language: str = "English",
            practice: Optional[dict] = None,
    ) -> None:
        self.lang = _lang_code(dialog_language)

        self.practice = practice or default_university_counseling_practice()

        default_followups = list(self.QUESTIONS)
        default_bg_questions = list(self.BACKGROUND_QUESTIONS)
        default_riasec_questions = list(self.RIASEC_QUESTIONS)

        self.QUESTIONS = get_practice_list(
            self.practice,
            "dialogue_config",
            "student",
            "fixed_followup_questions",
            dialog_language=dialog_language,
            default=default_followups,
        )

        self.BACKGROUND_QUESTIONS = get_practice_list(
            self.practice,
            "dialogue_config",
            "counselor",
            "background_questions",
            dialog_language=dialog_language,
            default=default_bg_questions,
        )

        self.RIASEC_QUESTIONS = get_practice_list(
            self.practice,
            "dialogue_config",
            "counselor",
            "riasec_questions",
            dialog_language=dialog_language,
            default=default_riasec_questions,
        )
        if self.lang == "it":
            self._OPTIONS_RX = re.compile(
                r"(?:\b(?:quale opzione vuoi scegliere|quale opzione vuoi perseguire)\b|^\s*(?:Opzione\s*1\b|1[\.\)]\s+)|\bScegli\s+1\b)",
                re.IGNORECASE | re.MULTILINE,
            )

            self._ADVICE_FALLBACK_RX = re.compile(
                r"(potresti chiarire un punto prima che ti consigli dei corsi|quale vincolo dovrei allentare per primo)",
                re.IGNORECASE,
            )

            self._GOODBYE_RX = re.compile(r"\b(arrivederci|ciao)\b", re.IGNORECASE)

        else:
            self._OPTIONS_RX = re.compile(
                r"(?:\bwhich option do you want to pursue\b|^\s*(?:Option\s*1\b|1[\.\)]\s+)|\bChoose\s+1\b)",
                re.IGNORECASE | re.MULTILINE,
            )

            self._ADVICE_FALLBACK_RX = re.compile(
                r"(could you clarify one point before i recommend programs|which constraint should i relax first)",
                re.IGNORECASE,
            )

            self._GOODBYE_RX = re.compile(r"\bgoodbye\b", re.IGNORECASE)

        self._started = False
        self._options_seen = False
        self._q_idx = 0
        self._done = False

    def reset(self) -> None:
        self._started = False
        self._options_seen = False
        self._q_idx = 0
        self._done = False
        clear_student_script_state()

    def _get_last_counselor_text(self, dialog, utterance: str) -> str:
        turns = getattr(dialog, "turns", None) or []
        last_counselor = (utterance or "").strip()

        for t in reversed(turns):
            txt = _safe_turn_text(t)
            if txt:
                last_counselor = txt
                break

        return last_counselor

    def _contains_any_question(self, text: str, questions: List[str]) -> bool:
        low = (text or "").strip().lower()
        return any(q.lower() in low for q in questions)

    def instruct(self, dialog, utterance):
        lang_phrase = "in Italian" if self.lang == "it" else "in English"
        turns = getattr(dialog, "turns", None) or []
        last_counselor = self._get_last_counselor_text(dialog, utterance)

        if self._done:
            set_student_script_state("goodbye_only", "Goodbye")
            return "STUDENT SCRIPT (HIGH PRIORITY): Output ONLY: Goodbye."

        if not self._started and len(turns) == 0:
            self._started = True
            clear_student_script_state()
            return (
                "STUDENT SCRIPT (HIGH PRIORITY): This is your FIRST message.\n"
                "- Present yourself as a student seeking guidance on choosing a university program.\n"
                "- Indicate your geographic location early (if available in your BACKGROUND).\n"
                "- If a first name is provided in your persona instructions, include it naturally.\n"
                "- If no first name is provided, do NOT invent one.\n"
                "- Convey gender only indirectly through name or natural grammar if present; do NOT state gender as a label.\n"
                "- Do NOT use metadata-like phrasing.\n"
                "- Use ONLY details consistent with your BACKGROUND and PERSONAL RULES.\n"
                "- For formal education, say only that you completed high school or a bachelor's degree, according to BACKGROUND.\n"
                "- Do NOT claim to be employed, a worker, a researcher, a master's graduate, or a PhD graduate.\n"
                "- 2-4 sentences, natural.\n"
            )
        rag_phase = ""
        try:
            rag_phase = str(get_last_rag_phase() or "").strip().lower()
        except Exception:
            rag_phase = ""
            
        if rag_phase == "done":
            self._done = True
            set_student_script_state("goodbye_only", "Goodbye")
            return (
                "STUDENT SCRIPT (HIGH PRIORITY): "
                "Output ONLY: Goodbye."
            )

        if rag_phase == "clarify_scope":
            set_student_script_state("answer_only_no_question")
            return (
                "STUDENT SCRIPT (HIGH PRIORITY):\n"
                "- The counselor has NOT proposed grounded options yet.\n"
                "- Answer ONLY the clarification question directly.\n"
                "- Do NOT choose an option.\n"
                "- Do NOT start the fixed follow-up sequence.\n"
            )

        if (not self._options_seen) and rag_phase == "advice":
            self._options_seen = True
            self._q_idx = 0
            set_student_script_state("choose_and_q1", self.QUESTIONS[0])
            return (
                "STUDENT SCRIPT (HIGH PRIORITY):\n"
                "- The counselor has presented grounded numbered options.\n"
                "- Choose ONE option clearly.\n"
                "- Then ask the following question VERBATIM on a new line:\n"
                f"{self.QUESTIONS[0]}\n"
                "- Output EXACTLY 2 lines: choice line + the question line.\n"
            )

        if self._options_seen:
            if self._q_idx < len(self.QUESTIONS) - 1:
                self._q_idx += 1
                q = self.QUESTIONS[self._q_idx]
                set_student_script_state("followup_only", q)
                return (
                    "STUDENT SCRIPT (HIGH PRIORITY):\n"
                    "- Ask the following question VERBATIM.\n"
                    "- Output ONLY that single line.\n"
                    f"{q}\n"
                )

            self._done = True
            set_student_script_state("thank_goodbye", "Thank you for your help, goodbye.")
            return "STUDENT SCRIPT (HIGH PRIORITY): Output ONLY: Thank you for your help, goodbye."

        # pre-options strict answer-only modes
        if self._contains_any_question(last_counselor, self.BACKGROUND_QUESTIONS):
            set_student_script_state("answer_only_no_question")
            return (
                "STUDENT SCRIPT (HIGH PRIORITY):\n"
                "- Reply as the student to the counselor's last message.\n"
                f"- Answer ONLY the counselor's question directly, {lang_phrase}.\n"
                "- Use ONLY details consistent with your BACKGROUND and PERSONAL RULES.\n"
                "- Be natural and specific (1–3 sentences).\n"
                "- Do NOT ask any question.\n"
                "- Do NOT restart the conversation.\n"
                "- Do NOT include any meta-commentary.\n"
            )

        if self._contains_any_question(last_counselor, self.RIASEC_QUESTIONS):
            set_student_script_state("answer_only_no_question")
            return (
                "STUDENT SCRIPT (HIGH PRIORITY):\n"
                "- Reply as the student to the counselor's last message.\n"
                f"- Answer ONLY the counselor's question directly, {lang_phrase}.\n"
                "- Use BACKGROUND and INTERESTS only when they provide evidence relevant to this specific activity.\n"
                "- Use PERSONALITY TRAITS only to shape tone and wording, not to decide whether the activity is liked.\n"
                "- Do not infer liking from gender, school type, job prestige, general competence, or answers to other RIASEC questions.\n"
                "- If direct evidence is absent or mixed, answer neutrally or express uncertainty.\n"
                "- Do not default to liking the activity; if it does not fit the profile, answer naturally with low or no interest.\n"
                "- Keep it brief and natural (1 sentence, or at most 2 short sentences).\n"
                "- Do NOT ask any question.\n"
                "- Do NOT restart the conversation.\n"
                "- Do NOT include any meta-commentary.\n"
            )

        if self._ADVICE_FALLBACK_RX.search(last_counselor or ""):
            set_student_script_state("answer_only_no_question")
            return (
                "STUDENT SCRIPT (HIGH PRIORITY):\n"
                "- Reply as the student to the counselor's last message.\n"
                f"- Answer ONLY the clarification question directly, {lang_phrase}.\n"
                "- Use ONLY details consistent with your BACKGROUND and PERSONAL RULES.\n"
                "- Be concise: 1-2 sentences max.\n"
                "- Do NOT ask any question.\n"
                "- Do NOT invent universities, deadlines, websites, locations, or facts.\n"
                "- Do NOT restart the conversation.\n"
                "- Do NOT include any meta-commentary.\n"
            )

        set_student_script_state("answer_only_no_question")
        return (
            "STUDENT SCRIPT (HIGH PRIORITY):\n"
            "- Reply as the student to the counselor's last message.\n"
           f"- Answer the counselor's question directly, {lang_phrase}.\n"
            "- Use ONLY details consistent with your BACKGROUND and PERSONAL RULES.\n"
            "- Be natural and specific (1–3 sentences).\n"
            "- Do NOT ask any question.\n"
            "- Do NOT restart the conversation.\n"
            "- Do NOT include any meta-commentary.\n"
        )

    def can_finish(self, utterance: str) -> bool:
        return self._done and bool(self._GOODBYE_RX.search(utterance or ""))
