# student_profile.py
from __future__ import annotations

import json
import re
import hashlib
from typing import Any, Dict, List

from .student import Student

from .annotation_compact import compact_annotation
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

# Token geografici rumorosi / incoerenti che non devono guidare il personaggio
_CONFLICTING_GEO_TOKENS = [
    "Midwest",
    "Mid-West",
    "Mideast",
    "Mid-East",
    "Northeast",
    "North-East",
    "Northwest",
    "North-West",
    "Southeast",
    "South-East",
    "Southwest",
    "South-West",
    "USA",
    "United States",
]

_BG_HAS_SCHOOL_RX = re.compile(
    r"\b(high\s*school|liceo|istituto\s+tecnico|istituto\s+professionale|diploma)\b",
    re.IGNORECASE,
)

# Pattern semplici per frasi geografiche che vogliamo rendere coerenti col JSON
_LOCATION_CLAUSE_PATTERNS = [
    re.compile(
        r"\b(?:i(?:'m| am)\s+from|i(?:'m| am)\s+based\s+in|i\s+live\s+in|i\s+grew\s+up\s+in|i\s+was\s+born\s+in)\s+[^.!?]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfrom\s+(?:a\s+|an\s+|the\s+)?(?:small\s+town|town|city|rural\s+county|county|area|region)?[^.!?]{0,80}",
        re.IGNORECASE,
    ),
]

def _lang_name(dialog_language: str = "English") -> str:
    return "Italian" if (dialog_language or "").strip().lower().startswith("it") else "English"

def _clean_json_region(value: Any) -> str:
    s = str(value or "").strip().title()
    return s if s in JSON_REGION_TO_BIO_PHRASE else ""


def _det_choice(seed: str, options: List[str]) -> str:
    h = hashlib.md5(seed.encode("utf-8", errors="ignore")).hexdigest()
    return options[int(h[:8], 16) % len(options)]


def _replace_conflicting_tokens(text: str, display_region: str) -> str:
    t = text
    for tok in _CONFLICTING_GEO_TOKENS:
        t = re.sub(rf"\b{re.escape(tok)}\b", display_region, t, flags=re.IGNORECASE)
    return t


def _normalize_location_clauses(text: str, display_region: str) -> str:
    t = text

    # Se ci sono frasi tipo "I'm from ...", "I grew up in ...", ecc.,
    # le rendiamo coerenti con la regione del JSON.
    replacements = [
        f"I am based in {display_region}",
        f"I am from {display_region}",
    ]

    for i, rx in enumerate(_LOCATION_CLAUSE_PATTERNS):
        repl = replacements[min(i, len(replacements) - 1)]
        t = rx.sub(repl, t)

    return t


def normalize_background_geography(text: str, *, authoritative_region: str = "") -> str:
    t = (text or "").strip()
    region = _clean_json_region(authoritative_region)

    if not region:
        return t

    display_region = JSON_REGION_TO_BIO_PHRASE[region]

    # 1) sostituisci token incoerenti tipo Midwest/USA
    t = _replace_conflicting_tokens(t, display_region)

    # 2) rendi coerenti eventuali clausole geografiche libere nella biography
    t = _normalize_location_clauses(t, display_region)

    # 3) aggiungi una frase finale autorevole, se non già presente
    low = t.lower()
    if display_region.lower() not in low:
        t = t.rstrip()
        if t and not t.endswith((".", "!", "?")):
            t += "."
        t += f" I am based in {display_region}."

    return t


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
        parts += ["Be energetic and curious; you can share brief reactions and ask follow-ups."]
    else:
        parts += ["Keep a balanced, conversational tone."]

    if level(emo_stab) == "low":
        parts += [
            "If you feel uncertain or anxious, show it briefly and ask for reassurance or a simple next step.",
            "Use mild hedging sometimes (e.g., 'I think', 'maybe', 'I'm not sure').",
        ]
    elif level(emo_stab) == "high":
        parts += ["Stay calm and pragmatic; focus on practical steps."]

    if level(consc) == "high":
        parts += ["You like structure: ask for steps, approximate timelines, and what to do first."]
    elif level(consc) == "low":
        parts += ["You prefer exploring options: ask for a few suggestions before committing to a plan."]

    if level(open_) == "low":
        parts += ["Prefer familiar/traditional options; ask about practical support if big changes feel stressful."]
    elif level(open_) == "high":
        parts += ["Be open to new experiences; ask about opportunities and growth."]

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
            desc.append("Realistic profile: prefers practical, hands-on activities and working with tools or objects.")
        elif code == "Investigative":
            desc.append("Investigative profile: enjoys understanding how things work, analysing problems, and logical subjects.")
        elif code == "Artistic":
            desc.append("Artistic profile: enjoys expressing ideas and feelings through creative activities like art or design.")
        elif code == "Social":
            desc.append("Social profile: likes helping, teaching, listening, and working closely with people.")
        elif code == "Enterprising":
            desc.append("Enterprising profile: enjoys leading, proposing ideas, persuading, and organising projects.")
        elif code == "Conventional":
            desc.append("Conventional profile: prefers organised settings, clear procedures, and keeping things orderly.")
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
    gender = str(features.get("assigned_gender", "") or "").strip()
    region = _clean_json_region(features.get("region"))

    background = (
        normalize_background_geography(
            compact_annotation(annotation),
            authoritative_region=region,
        )
        if annotation else ""
    )

    if background and not background.endswith((".", "!", "?")):
        background = background.rstrip() + "."

    # Se manca una riga sulla scuola, la aggiungiamo usando SEMPRE la regione del JSON
    if not _BG_HAS_SCHOOL_RX.search(background or ""):
        seed = f"{annotation}|{region}|{age}|{gender}"
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
        bg_region = JSON_REGION_TO_BIO_PHRASE.get(region, "Italy")
        background = (background + " " if background else "") + (
            f"I completed high school at a {sector} {hs_type} in {bg_region}."
        )

    interests = riasec_to_interests_descriptions(gpt_riasec)
    personality_traits = bfi_to_personality_traits(bfi)
    student_rules = bfi_to_student_rules(bfi, dialog_language)

    return {
        "age": age,
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