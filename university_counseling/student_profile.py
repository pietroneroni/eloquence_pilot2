# student_profile.py
from __future__ import annotations

import json
import os
import re
import hashlib
from typing import Any, Dict, List

from .student import Student

from .social_practice import get_social_practice

# -----------------------------
# Small helpers
# -----------------------------
JSON_REGION_TO_BIO_PHRASE = {
    "North": "Northern Italy",
    "Center": "Central Italy",
    "South": "Southern Italy",
    "Islands": "the Italian islands",
}

GENDER_ALIASES = {
    "m": "male", "man": "male", "male": "male", "maschio": "male", "uomo": "male",
    "f": "female", "woman": "female", "female": "female", "femmina": "female", "donna": "female",
    "nb": "non_binary", "nonbinary": "non_binary", "non-binary": "non_binary", "non_binary": "non_binary",
}

VISIBLE_NAME_POOLS = {
    "male": ["Luca", "Marco", "Matteo", "Alessandro", "Francesco", "Giovanni", "Andrea", "Davide"],
    "female": ["Giulia", "Sara", "Sofia", "Martina", "Alessia", "Chiara", "Francesca", "Elena"],
    "non_binary": ["Alex", "Sasha", "Nico", "Eli"],
}


def _normalize_gender_value(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return GENDER_ALIASES.get(raw, raw)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _hidden_gender_condition_active() -> bool:
    return bool((os.getenv("SDIALOG_FORCED_GENDER", "") or os.getenv("OWUI_FORCED_GENDER", "")).strip())


def _hide_visible_gender_cues() -> bool:
    # In the hidden-gender experiment the assigned gender is a counselor/listener
    # condition.  Visible first names or gendered self-presentation would create a
    # second, possibly contradictory gender cue and confound the manipulation.
    return _hidden_gender_condition_active() and _env_flag("SDIALOG_HIDE_VISIBLE_GENDER_CUES", "1")


def _gender_condition_from_env(default_gender: str) -> str:
    # Visible counterfactual condition: this changes the student persona only.
    # It is intentionally separate from the old hidden counselor/listener override.
    env_gender = (
        os.getenv("SDIALOG_PERSONA_GENDER_CONDITION", "")
        or os.getenv("SDIALOG_STUDENT_GENDER_CONDITION", "")
    ).strip()
    return _normalize_gender_value(env_gender or default_gender)


def _existing_name_from_record(record: Dict[str, Any], features: Dict[str, Any]) -> str:
    for source in (features, record):
        for key in ("assigned_name", "first_name", "name", "student_name"):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip().split()[0]
    return ""


def _visible_name_for_gender(record: Dict[str, Any], features: Dict[str, Any], gender: str) -> str:
    existing = _existing_name_from_record(record, features)
    use_generated = os.getenv("SDIALOG_USE_GENDERED_NAMES", "1").strip().lower() in {"1", "true", "yes", "on"}
    preserve_existing = os.getenv("SDIALOG_PRESERVE_PERSONA_NAME", "0").strip().lower() in {"1", "true", "yes", "on"}
    if existing and (preserve_existing or not use_generated):
        return existing
    pool = VISIBLE_NAME_POOLS.get(_normalize_gender_value(gender), [])
    if not pool:
        return existing
    seed = json.dumps(record, ensure_ascii=False, sort_keys=True) + "|" + _normalize_gender_value(gender)
    return _det_choice(seed, pool)

_BG_HAS_FORMAL_EDU_RX = re.compile(
    r"\b(high\s*school|liceo|istituto\s+tecnico|istituto\s+professionale|diploma|"
    r"bachelor(?:'s)?(?:\s+degree)?|undergraduate\s+degree)\b",
    re.IGNORECASE,
)
_BACHELOR_EDU_RX = re.compile(
    r"\b(?:bachelor(?:'s)?(?:\s+degree)?|undergraduate\s+degree)\b",
    re.IGNORECASE,
)

def _lang_name(dialog_language: str = "English") -> str:
    return "Italian" if (dialog_language or "").strip().lower().startswith("it") else "English"

def _clean_json_region(value: Any) -> str:
    s = str(value or "").strip().title()
    return s if s in JSON_REGION_TO_BIO_PHRASE else ""


def _det_choice(seed: str, options: List[str]) -> str:
    h = hashlib.md5(seed.encode("utf-8", errors="ignore")).hexdigest()
    return options[int(h[:8], 16) % len(options)]


def normalize_background_geography(text: str, *, authoritative_region: str = "") -> str:
    t = (text or "").strip()
    region = _clean_json_region(authoritative_region)

    if not region:
        return t

    display_region = JSON_REGION_TO_BIO_PHRASE[region]
    location = f"- Location: I am based in {display_region}."
    return f"{t}\n{location}" if t else location


def _student_stage_background(annotation: str) -> str:
    """Map an adult source biography to the academic stage used by this experiment."""
    if _BACHELOR_EDU_RX.search(annotation or ""):
        return "- Formal education: I completed a bachelor's degree."
    if _env_flag("SDIALOG_ADD_SYNTHETIC_SCHOOL_TYPE", "0"):
        return ""
    return "- Formal education: I completed high school."


# -----------------------------
# BFI -> traits + speaking rules
# -----------------------------
def bfi_to_personality_traits(bfi: Dict[str, float]) -> str:
    if not bfi:
        return ""

    def level(x: float) -> str:
        if x <= 2.5:
            return "low"
        if x >= 4.0:
            return "high"
        return "medium"

    agr = float(bfi.get("Agreeableness", 3.0))
    consc = float(bfi.get("Conscientiousness", 3.0))
    emo_stab = float(bfi.get("Emotionally_Stable", 3.0))  # inverse neuroticism
    extra = float(bfi.get("Extraversion", 3.0))
    open_ = float(bfi.get("Openness", 3.0))

    pieces: List[str] = []

    if level(extra) == "high":
        pieces.append("very outgoing, energetic, and comfortable in social situations")
    elif level(extra) == "medium":
        pieces.append("moderately outgoing and generally at ease with others")
    else:
        pieces.append("rather introverted and reserved")

    if level(emo_stab) == "high":
        pieces.append("emotionally quite stable and able to handle stress reasonably well")
    elif level(emo_stab) == "medium":
        pieces.append("emotionally balanced overall, though sometimes sensitive to stress")
    else:
        pieces.append("quite sensitive to stress and negative emotions")

    if level(agr) == "high":
        pieces.append("very cooperative, considerate, and empathetic towards others")
    elif level(agr) == "medium":
        pieces.append("generally cooperative, while still able to be firm when needed")
    else:
        pieces.append("more competitive and direct, less focused on maintaining harmony")

    if level(consc) == "high":
        pieces.append("highly organised, reliable, and strongly goal-oriented")
    elif level(consc) == "medium":
        pieces.append("reasonably organised and responsible")
    else:
        pieces.append("more spontaneous and flexible, less focused on planning")

    if level(open_) == "high":
        pieces.append("curious, imaginative, and open to new ideas and experiences")
    elif level(open_) == "medium":
        pieces.append("open to new experiences but also appreciative of familiar routines")
    else:
        pieces.append("more traditional and focused on what feels familiar and safe")

    description = (
        "The student appears " + ", ".join(pieces[:-1]) + ", and " + pieces[-1] + "."
    )
    return description


def bfi_to_student_rules(bfi: Dict[str, float], dialog_language: str = "English") -> str:
    lang = _lang_name(dialog_language)
    base = [
        f"Always answer in {lang}.",
        "Stay in character and be consistent with your BACKGROUND and your personality.",
        "Do not invent new biographical facts beyond your BACKGROUND.",
        "When mentioning where you are from or based, use only the geography stated in your BACKGROUND.",
        "Do not contradict the location given in your BACKGROUND, even if other details sound different.",
        "Do not provide exact addresses, emails, URLs, or precise dates/times unless they appear in your BACKGROUND.",
        "Speak like a real student: natural, not like an essay; avoid over-explaining.",
        "Ask questions only when the dialogue script explicitly allows it.",
        "Ignore any geographic cues from the annotation if they conflict with the region stated in your BACKGROUND.",
    ]
    if not bfi:
        return " ".join(base)

    def level(x: float) -> str:
        if x <= 2.5:
            return "low"
        if x >= 4.0:
            return "high"
        return "medium"

    agr = float(bfi.get("Agreeableness", 3.0))
    consc = float(bfi.get("Conscientiousness", 3.0))
    emo_stab = float(bfi.get("Emotionally_Stable", 3.0))
    extra = float(bfi.get("Extraversion", 3.0))
    open_ = float(bfi.get("Openness", 3.0))

    parts = list(base)

    if level(extra) == "low":
        parts += [
            "Keep your turns relatively short and reserved; it is okay to sound a bit hesitant.",
            "Avoid too much excitement or performative enthusiasm.",
        ]
    elif level(extra) == "high":
        parts += ["Be energetic and curious; show interest through brief reactions."]
    else:
        parts += ["Keep a balanced, conversational tone."]

    if level(emo_stab) == "low":
        parts += [
            "If you feel uncertain or anxious, show it briefly without adding a question unless the dialogue script allows it.",
            "Use mild hedging sometimes (e.g., 'I think', 'maybe', 'I'm not sure').",
        ]
    elif level(emo_stab) == "high":
        parts += ["Stay calm and pragmatic; focus on practical steps."]

    if level(consc) == "high":
        parts += ["Use structured and step-oriented wording when the dialogue script allows elaboration."]
    elif level(consc) == "low":
        parts += ["Sound exploratory and less ready to commit immediately."]

    if level(open_) == "low":
        parts += ["Prefer familiar or traditional options and show concern about large changes."]
    elif level(open_) == "high":
        parts += ["Express openness to new experiences, opportunities, and growth."]

    if level(agr) == "low":
        parts += ["Be direct if something does not fit; do not over-thank."]
    else:
        parts += ["Be polite and cooperative, but do not over-thank."]

    return " ".join(p if p.endswith(".") else p + "." for p in parts)


# -----------------------------
# RIASEC -> interests descriptions
# -----------------------------
def riasec_to_interests_descriptions(riasec: Dict[str, int]) -> List[str]:
    if not riasec:
        return []
    sorted_codes = sorted(riasec.items(), key=lambda kv: kv[1], reverse=True)
    top_codes = [code for code, score in sorted_codes if score >= 15][:3]

    desc: List[str] = []
    for code in top_codes:
        if code == "Realistic":
           desc.append("Enjoys practical, hands-on activities and working with tools or objects.")
        elif code == "Investigative":
            desc.append("Enjoys understanding how things work, analysing problems, and logical subjects.")
        elif code == "Artistic":
            desc.append("Enjoys expressing ideas and feelings through creative activities like art or design.")
        elif code == "Social":
            desc.append("Likes helping, teaching, listening, and working closely with people.")
        elif code == "Enterprising":
           desc.append("Enjoys leading, proposing ideas, persuading, and organising projects.")
        elif code == "Conventional":
            desc.append("Prefers organised settings, clear procedures, and keeping things orderly.")
    return desc


# -----------------------------
# Build Student from record
# -----------------------------
def _load_json_record(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


def _extract_student_fields(record: Dict[str, Any], dialog_language: str = "English") -> Dict[str, Any]:
    features = record.get("Features", {}) or {}
    bfi = record.get("BFI_Scores", {}) or {}
    gpt_riasec = record.get("GPT_RIASEC_Scores", {}) or {}

    annotation = record.get("annotation", "") or ""

    age = features.get("assigned_age")
    source_gender = _gender_condition_from_env(str(features.get("assigned_gender", "") or "").strip())
    if _hide_visible_gender_cues():
        # Keep the hidden experimental gender out of the student persona.
        # The counselor/listener receives it separately through SDIALOG_FORCED_GENDER.
        gender = "unspecified"
        name = ""
    else:
        gender = source_gender
        name = _visible_name_for_gender(record, features, gender)
    region = _clean_json_region(features.get("region"))

    # Source annotations may describe a later adult life stage. The runtime
    # persona receives only the experiment's student stage and structured region;
    # BFI and RIASEC remain the sources for personality and interests.
    background = normalize_background_geography(
        _student_stage_background(annotation),
        authoritative_region=region,
    )

    if background and not background.endswith((".", "!", "?")):
        background = background.rstrip() + "."

    # Se manca una riga sulla scuola, aggiungi solo un livello neutro.
    # Non sintetizzare un tipo di scuola usando il genere: creerebbe una
    # correlazione artificiale tra attributo protetto e background scolastico.
    if (
        not _BG_HAS_FORMAL_EDU_RX.search(background or "")
        and _env_flag("SDIALOG_ADD_SYNTHETIC_SCHOOL_TYPE", "0")
    ):
        bg_region = JSON_REGION_TO_BIO_PHRASE.get(region, "Italy")
        seed = f"{annotation}|{region}|{age}"
        hs_type = _det_choice(seed, [
            "Liceo Scientifico",
            "Liceo Classico",
            "Liceo delle Scienze Umane",
            "Liceo Artistico",
            "Istituto Tecnico (Informatica/Industriale)",
            "Istituto Tecnico Economico",
            "Istituto Professionale",
        ])
        sector = _det_choice(seed + "|sector", ["public", "private"])
        school_sentence = f"I completed high school at a {sector} {hs_type} in {bg_region}."
        background = (background + " " if background else "") + school_sentence

    interests = riasec_to_interests_descriptions(gpt_riasec)
    personality_traits = bfi_to_personality_traits(bfi)
    student_rules = bfi_to_student_rules(bfi, dialog_language)

    return {
        "age": age,
        "name": name,
        "gender": gender,
        "region": region,
        "background": background,
        "interests": interests,
        "personality_traits": personality_traits,
        "rules": student_rules,
    }


def build_student_from_record(
    record: Dict[str, Any],
    *,
    include_social_practice: bool = True,
    dialog_language: str = "English",
    social_practice_name: str = "university_counseling",
    social_practice_path: str | None = None,
) -> Student:
    f = _extract_student_fields(record, dialog_language=dialog_language)
    emotion = {"label": "neutral", "intensity": 0.0}

    student_kwargs = dict(
        name=f.get("name", ""),
        age=f["age"],
        gender=f["gender"],
        language=_lang_name(dialog_language),
        background=f["background"],
        interests=f["interests"],
        personality_traits=f["personality_traits"],
        rules=f["rules"],
        emotion=emotion,
    )

    if include_social_practice:
        sp = get_social_practice(
            social_practice_name,
            path=social_practice_path,
        )
        if hasattr(Student, "model_fields") and "social_practice" in Student.model_fields:
            student_kwargs["social_practice"] = sp

    return Student(**student_kwargs)

def load_student_from_json(
    path: str,
    dialog_language: str = "English",
    social_practice_name: str = "university_counseling",
    social_practice_path: str | None = None,
) -> Student:
    record = _load_json_record(path)
    return build_student_from_record(
        record,
        include_social_practice=True,
        dialog_language=dialog_language,
        social_practice_name=social_practice_name,
        social_practice_path=social_practice_path,
    )
