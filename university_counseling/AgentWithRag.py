from __future__ import annotations

import copy
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from sdialog.orchestrators import BaseOrchestrator

from university_counseling.clean_options import clean_and_dedup_options
from collections import defaultdict

from university_counseling.social_practice import (
    default_university_counseling_practice,
    get_practice_list,
)

# =============================================================================
# Constants
# =============================================================================
ZONE_TO_UNIVERSITIES = {
    "north": [
        "Università degli studi di Trento",
        "Università di Trento",
        "University of Trento",
    ],
    "center": [
        "Università degli Studi dell'Aquila",
        "Università dell'Aquila",
        "University of L'Aquila",
        "Università degli Studi G. d'Annunzio Chieti-Pescara",
        "Università degli Studi 'Gabriele d'Annunzio' Chieti-Pescara",
        "Università degli Studi di Chieti-Pescara",
        "Gabriele d'Annunzio University of Chieti-Pescara",
        "Università degli Studi di Siena",
        "Università di Siena",
        "University of Siena",
    ],
    "south": [
        "Università degli Studi di Foggia",
        "Università di Foggia",
        "University of Foggia",
        "Università degli Studi di Salerno",
        "Università di Salerno",
        "University of Salerno",
    ],
    "islands": [
        "Università degli Studi di Messina",
        "Università di Messina",
        "University of Messina",
        "Università degli Studi di Sassari",
        "Università di Sassari",
        "University of Sassari",
    ],
}

ZONE_CHAINS = {
    "north": ["north", "center", "south", "islands"],
    "center": ["center", "north", "south", "islands"],
    "south": ["south", "center", "islands", "north"],
    "islands": ["islands", "south", "center", "north"],
}

FIELD_OF_INTEREST_LABELS = (
    "arts_design",
    "communication_digital_media",
    "cultural_heritage",
    "humanities",
    "psychology",
    "cognitive_science_linguistics",
    "social_sciences",
    "life_sciences",
    "environmental_science",
    "environmental_engineering",
    "computer_science",
    "engineering_technology",
    "physical_sciences",
    "health_technology",
)

FIELD_QUERY_TERMS = {
    "arts_design": [
        "art and design",
        "arts design",
        "visual arts",
        "arti visive",
        "design",
        "grafica",
        "creative design",
    ],
    "communication_digital_media": [
        "digital communication",
        "digital media",
        "communication design",
        "comunicazione digitale",
        "comunicazione e media",
        "media digitali",
    ],
    "cultural_heritage": [
        "cultural heritage",
        "patrimonio culturale",
        "beni culturali",
        "art history",
        "storia dell'arte",
        "archaeology",
        "archeologia",
        "museums",
        "musei",
    ],
    "humanities": [
        "humanities",
        "scienze umanistiche",
        "lettere",
        "literature",
        "letteratura",
        "philosophy",
        "filosofia",
        "languages",
        "lingue",
        "classics",
        "studi classici",
    ],
    "psychology": [
        "psychology",
        "psicologia",
        "clinical psychology",
        "psicologia clinica",
        "counseling",
        "mental health",
        "salute mentale",
        "psychotherapy",
        "psicoterapia",
    ],
    "cognitive_science_linguistics": [
        "cognitive science",
        "scienze cognitive",
        "linguistics",
        "linguistica",
        "mind and language",
        "mente e linguaggio",
        "neuroscience",
        "neuroscienze",
    ],
    "social_sciences": [
        "social sciences",
        "scienze sociali",
        "sociology",
        "sociologia",
        "political studies",
        "scienze politiche",
        "community studies",
    ],
    "life_sciences": [
        "biology",
        "biologia",
        "biological sciences",
        "scienze biologiche",
        "biotechnology",
        "biotecnologie",
        "life sciences",
        "scienze della vita",
        "conservation biology",
    ],
    "environmental_science": [
        "environmental science",
        "scienze ambientali",
        "ecology",
        "ecologia",
        "conservation",
        "conservazione",
        "sustainability",
        "sostenibilità",
        "ecosystems",
        "ecosistemi",
    ],
    "environmental_engineering": [
        "environmental engineering",
        "ingegneria ambientale",
        "engineering for the environment",
        "ingegneria per l'ambiente",
        "ambiente e territorio",
        "environmental technology",
        "tecnologia ambientale",
    ],
    "computer_science": [
        "computer science",
        "informatica",
        "software",
        "data systems",
        "data analysis",
        "GIS",
        "artificial intelligence",
        "intelligenza artificiale",
    ],
    "engineering_technology": [
        "engineering",
        "ingegneria",
        "technology",
        "tecnologia",
        "mechanical engineering",
        "civil engineering",
        "electronics",
        "mechatronics",
        "industrial systems",
    ],
    "physical_sciences": [
        "physics",
        "fisica",
        "chemistry",
        "chimica",
        "mathematics",
        "matematica",
        "physical sciences",
        "scienze fisiche",
        "applied sciences",
        "scienze applicate",
    ],
    "health_technology": [
        "health technology",
        "healthcare technology",
        "biomedical technology",
        "tecnologie sanitarie",
        "medical technology",
        "neurophysiopathology",
        "neurofisiopatologia",
        "rehabilitation",
        "riabilitazione",
    ],
}


KEYWORD_FALLBACK_TERMS = {
    "arts_design": ["design", "art", "arts", "visual", "creative", "graphics", "architecture"],
    "communication_digital_media": ["communication", "media", "digital", "interfaces", "data visualization"],
    "cultural_heritage": ["cultural heritage", "heritage", "art history", "archaeology", "museum"],
    "humanities": ["philology", "literature", "languages", "history", "philosophy"],
    "psychology": ["psychology", "clinical psychology", "counseling", "mental health", "behaviour"],
    "cognitive_science_linguistics": ["cognitive", "linguistics", "language", "mind", "neuroscience"],
    "social_sciences": ["social", "sociology", "education", "political", "community"],
    "life_sciences": ["biotechnology", "biotechnologies", "biology", "biological", "biomolecular", "molecular", "cellular"],
    "environmental_science": ["environmental", "environment", "conservation", "biodiversity", "ecology", "ecotoxicology", "natural sciences", "sustainability"],
    "environmental_engineering": ["environmental engineering", "environmental", "territorial engineering", "territory", "sustainability"],
    "computer_science": ["computer science", "computer", "software", "data science", "artificial intelligence", "informatics"],
    "engineering_technology": ["engineering", "computer engineering", "electronic engineering", "civil engineering", "mechanical", "mechatronics", "industrial"],
    "physical_sciences": ["physics", "chemistry", "mathematics", "applied sciences", "planetary"],
    "health_technology": ["biomedical", "health", "medical", "rehabilitation", "neurophysiopathology"],
}


def _field_label_list_for_prompt() -> str:
    return "\n".join(f"  {label}" for label in FIELD_OF_INTEREST_LABELS)


FIELD_OF_INTEREST_RULES = (
    "Rules for inferred.field_of_interest:\n"
    "- Return field_of_interest as an ordered JSON array of 1 to 3 labels.\n"
    "- Labels must be distinct and ordered from strongest to weakest fit.\n"
    "- Choose labels only from this list:\n"
    f"{_field_label_list_for_prompt()}\n"
    "- Pick the intended university area, not the current job.\n"
    "- Prefer concrete course-family areas over abstract hybrid concepts.\n"
    "- Visual art, art practice, design, product/graphic/visual design -> arts_design.\n"
    "- Digital communication, media, communication design, online/media production -> communication_digital_media.\n"
    "- Art history, archaeology, museums, cultural assets, heritage tourism -> cultural_heritage.\n"
    "- Literature, philosophy, languages, classics, history as humanities -> humanities.\n"
    "- Psychology, counseling, psychotherapy, mental health, clinical behavior -> psychology.\n"
    "- Cognitive science, linguistics, mind/language, computational cognition -> cognitive_science_linguistics.\n"
    "- Sociology, political/social studies, international or community social studies -> social_sciences.\n"
    "- Biology, biotechnology, life sciences, lab biology -> life_sciences.\n"
    "- Ecology, conservation, sustainability, ecosystems, nature reserves -> environmental_science.\n"
    "- Environmental engineering, environmental technology, technical conservation systems, environment/territory engineering -> environmental_engineering.\n"
    "- Informatics, computer science, software, AI, data systems, GIS/data tech -> computer_science.\n"
    "- Engineering, mechanics, mechatronics, electronics, industrial/civil/technical systems -> engineering_technology.\n"
    "- Physics, chemistry, mathematics, laboratory science, broad applied/physical sciences -> physical_sciences.\n"
    "- Biomedical/healthcare technology, rehabilitation techniques, neurophysiopathology, medical technology -> health_technology.\n"
    "- If a profile mixes tech + environment, return environmental_engineering first if the student wants technical systems; otherwise environmental_science first.\n"
    "- If a profile mixes biology/biotech + environment, return life_sciences first if biotech/lab biology is central; otherwise environmental_science first.\n"
    "- If a profile mixes psychology + cognition/linguistics, return psychology first for clinical/helping goals; return cognitive_science_linguistics first for mind/language/research goals.\n"
    "- If only one label is clearly supported, return a one-item array.\n"
    "- If no label is clearly supported, omit the field.\n"
    "- Do not invent a label outside the list.\n"
)

REGION_TO_RAG_MACROAREA = {
    "north": "Nord",
    "center": "Centro",
    "south": "Sud",
    "islands": "Isole",
}

CHOICE_REPLY_OPTIONS = {
    "en": {
        1: ("focusing on a technical program", "a program oriented toward humanities"),
        4: ("take a leadership role now", "focus on supporting tasks first"),
        7: ("take advanced classes immediately", "start with introductory modules"),
    },
    "it": {
        1: ("concentrarti su un percorso tecnico", "un percorso orientato alle discipline umanistiche"),
        4: ("assumere già ora un ruolo di leadership", "concentrarti prima su compiti di supporto"),
        7: ("iniziare subito con corsi avanzati", "partire da moduli introduttivi"),
    },
}

_OPT_RX = re.compile(r"\b(?:option|opzione)\s*([1-3])\b", re.I)

_FLOW_YES_RX = re.compile(
    r"\b(yes|yeah|yep|sure|definitely|absolutely|sì|si|certo|assolutamente)\b",
    re.I,
)

_FLOW_NO_RX = re.compile(
    r"\b(no|nope|not really|don't|do not|cannot|can't|never|assolutamente no|direi di no)\b",
    re.I,
)
_NUM_ONLY_RX = re.compile(r"^\s*([1-3])\b")

_ENTITY_RX = re.compile(
    r"\b("
    r"Universit[aà]\s+[^,\n\.\|]{2,120}|"
    r"University\s+of\s+[^,\n\.]{2,120}|"
    r"Accademia\s+[^,\n\.]{2,120}|"
    r"Politecnico\s+[^,\n\.]{2,120}|"
    r"Conservatorio\s+[^,\n\.]{2,120}|"
    r"Istituto\s+[^,\n\.]{2,120}"
    r")\b",
    re.IGNORECASE,
)

_CLOSE_RX = re.compile(r"\b(thanks|thank you|bye|good luck|take care|goodbye)\b", re.I)

_LISTENER_PATCH_BLOCK_RX = re.compile(
    r"<LISTENER_PATCH>\s*(\{[\s\S]*?\})\s*</LISTENER_PATCH>",
    re.IGNORECASE,
)

_RIASEC_ALLOWED_LABELS = {
    "realistic": "Realistic",
    "investigative": "Investigative",
    "artistic": "Artistic",
    "social": "Social",
    "enterprising": "Enterprising",
    "conventional": "Conventional",
}

_RIASEC_LABEL_ORDER = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]

_RIASEC_WEIGHTS_BY_Q: List[Dict[str, float]] = [
    {"Realistic": 1.0},
    {"Investigative": 0.75, "Realistic": 0.25},
    {"Investigative": 1.0},
    {"Artistic": 1.0},
    {"Artistic": 0.45, "Investigative": 0.35, "Enterprising": 0.20},
    {"Social": 1.0},
    {"Enterprising": 0.75, "Conventional": 0.25},
    {"Conventional": 1.0},
]

def _normalize_riasec_valences(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list) or len(value) != 8:
        return None

    out: List[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            v = int(item)
        except Exception:
            return None
        if v < -2 or v > 2:
            return None
        out.append(v)

    return out

def _compute_riasec_normalized_scores(valences: List[int]) -> Dict[str, float]:
    scores = {label: 0.0 for label in _RIASEC_LABEL_ORDER}
    max_abs = {label: 0.0 for label in _RIASEC_LABEL_ORDER}

    for idx, valence in enumerate(valences):
        for label, weight in _RIASEC_WEIGHTS_BY_Q[idx].items():
            scores[label] += float(valence) * weight
            max_abs[label] += 2.0 * weight

    return {
        label: round(scores[label] / max_abs[label], 3) if max_abs[label] else 0.0
        for label in _RIASEC_LABEL_ORDER
    }


def _riasec_differentiation(scores: Dict[str, float]) -> float:
    vals = sorted(scores.values(), reverse=True)
    return round(vals[0] - vals[-1], 3)

def _riasec_top3_and_confidence_from_valences(valences: List[int]) -> Tuple[List[str], str]:
    scores = _compute_riasec_normalized_scores(valences)
    differentiation = _riasec_differentiation(scores)

    ordered = sorted(
        _RIASEC_LABEL_ORDER,
        key=lambda label: (-scores[label], _RIASEC_LABEL_ORDER.index(label)),
    )

    top3 = ordered[:3]

    top_score = scores[top3[0]]
    third = scores[top3[2]]
    fourth = scores[ordered[3]]
    third_fourth_gap = abs(third - fourth)

    if top_score <= 0.1 or third_fourth_gap < 0.10 or differentiation < 0.35:
        confidence = "low"
    elif top_score >= 0.55 and third_fourth_gap >= 0.20 and differentiation >= 0.60:
        confidence = "high"
    else:
        confidence = "medium"

    return top3, confidence



_REQ_RXES: List[Tuple[re.Pattern, List[str]]] = [
    (
        re.compile(r"\b(tuition|fees?|costs?|scholarship|grant)\b", re.I),
        ["tuition", "fees", "cost", "tasse", "contributi", "borsa", "scholarship", "grant"],
    ),
    (
        re.compile(r"\b(deadline|apply|admission|enroll)\b", re.I),
        ["deadline", "apply", "admission", "bando", "scadenza", "ammissione", "immatricolazione"],
    ),
]

def _lang_code(dialog_language: str = "English") -> str:
    return "it" if (dialog_language or "").strip().lower().startswith("it") else "en"

_DIALOG_LANGUAGE = "en"

def set_dialog_language(dialog_language: str = "English") -> None:
    global _DIALOG_LANGUAGE
    _DIALOG_LANGUAGE = _lang_code(dialog_language)

def _t(en: str, it: str) -> str:
    return it if _DIALOG_LANGUAGE == "it" else en

def _require_practice_list(
    practice: Dict[str, Any],
    *path: str,
    dialog_language: str = "English",
    min_len: int = 1,
    exact_len: Optional[int] = None,
) -> List[str]:
    values = get_practice_list(
        practice,
        *path,
        dialog_language=dialog_language,
        default=[],
    )

    dotted_path = ".".join(path)

    if exact_len is not None and len(values) != exact_len:
        raise ValueError(
            f"Invalid social_practices config: '{dotted_path}' for "
            f"{dialog_language!r} must contain exactly {exact_len} items, "
            f"found {len(values)}."
        )

    if len(values) < min_len:
        raise ValueError(
            f"Invalid social_practices config: '{dotted_path}' for "
            f"{dialog_language!r} must contain at least {min_len} item(s), "
            f"found {len(values)}."
        )

    return values

# =============================================================================
# Generic helpers
# =============================================================================
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().casefold()


def _ascii_norm(s: str) -> str:
    """Lowercase, strip accents and collapse whitespace for robust token matching."""
    t = unicodedata.normalize("NFKD", s or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.casefold()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _keyword_in_normalized_text(text: str, keyword: str) -> bool:
    """Token/phrase match; avoids substring false positives.

    Examples: technology does NOT match biotechnology; art does NOT match artificial.
    A few intentionally stemmed Italian/English roots are allowed.
    """
    t = _ascii_norm(text)
    k = _ascii_norm(keyword)
    if not t or not k:
        return False

    if " " in k:
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", t))

    tokens = set(t.split())
    variants = {k}
    if k.endswith("y") and len(k) > 2:
        variants.add(k[:-1] + "ies")
    else:
        variants.add(k + "s")
        variants.add(k + "es")

    intentional_stems = {
        "sostenibil", "sanitar", "biomed", "riabilit", "ambiental",
        "ecolog", "biotecnolog", "biolog", "informatic", "ingegner",
        "psicolog", "linguistic", "cognitiv", "artist", "progett",
    }
    if k in intentional_stems:
        return any(tok.startswith(k) for tok in tokens)

    return any(v in tokens for v in variants)


def _contains_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    return any(_keyword_in_normalized_text(text, kw) for kw in (keywords or []))

def _dprint(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg)


def _normalize_region_value(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None

    s = _normalize(value)

    aliases = {
        "north": "north",
        "nord": "north",
        "northern italy": "north",

        "center": "center",
        "centre": "center",
        "centro": "center",
        "central italy": "center",

        "south": "south",
        "sud": "south",
        "southern italy": "south",

        "islands": "islands",
        "isole": "islands",
        "the italian islands": "islands",
    }

    return aliases.get(s)


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


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _ctx_has_any(ctx_norm: str, keys: List[str]) -> bool:
    return any(k and (k.lower() in ctx_norm) for k in keys)


def _all_known_universities() -> List[str]:
    return _flatten_to_strings(list(ZONE_TO_UNIVERSITIES.values()))


def _build_university_chain(preferred_zone: Optional[str]) -> List[Tuple[str, List[str]]]:
    zone = _normalize_region_value(preferred_zone)
    if not zone or zone not in ZONE_CHAINS:
        return [("all", _all_known_universities())]

    out: List[Tuple[str, List[str]]] = []
    for z in ZONE_CHAINS[zone]:
        out.append((z, list(ZONE_TO_UNIVERSITIES.get(z, []))))
    return out


def _flatten_to_strings(values: Any) -> List[str]:
    out: List[str] = []

    def _walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
            return
        if isinstance(v, (list, tuple, set)):
            for item in v:
                _walk(item)
            return
        s = str(v).strip()
        if s:
            out.append(s)

    _walk(values)
    return out


def _university_matches_allowed(university: str, allowed_universities: Optional[Sequence[str]]) -> bool:
    if not allowed_universities:
        return True

    u_norm = _normalize(university)
    flat_allowed = _flatten_to_strings(allowed_universities)

    for allowed in flat_allowed:
        a_norm = _normalize(allowed)
        if not a_norm:
            continue
        if u_norm == a_norm or u_norm in a_norm or a_norm in u_norm:
            return True

    return False






def _infer_zone_from_university(university: Optional[str]) -> Optional[str]:
    if not isinstance(university, str) or not university.strip():
        return None

    target = _normalize(university)
    for zone, names in ZONE_TO_UNIVERSITIES.items():
        for name in names:
            if _normalize(name) == target:
                return zone
    return None


def _parse_program_block(block: str) -> Tuple[str, str, str]:
    university = ""
    course = ""
    tipo = ""
    for line in (block or "").splitlines():
        line = line.strip()
        if line.startswith("UNIVERSITÀ:"):
            university = line.split(":", 1)[1].strip()
        elif line.startswith("CORSO:"):
            course = line.split(":", 1)[1].strip()
        elif line.startswith("TIPO:"):
            tipo = line.split(":", 1)[1].strip()
    return university, course, tipo


def _option_key(option: str) -> str:
    return _normalize(option)


# =============================================================================
# RIASEC helpers
# =============================================================================
def _normalize_riasec_labels(value: Any) -> Optional[List[str]]:
    if not isinstance(value, (list, tuple)):
        return None

    out: List[str] = []
    seen = set()

    for item in value:
        if not isinstance(item, str):
            return None
        key = item.strip().lower()
        if key not in _RIASEC_ALLOWED_LABELS:
            return None
        canon = _RIASEC_ALLOWED_LABELS[key]
        if canon not in seen:
            seen.add(canon)
            out.append(canon)

    if len(out) != 3:
        return None

    return out


def _riasec_labels_to_code(labels: Any) -> Optional[str]:
    if not isinstance(labels, (list, tuple)) or len(labels) != 3:
        return None

    mapping = {
        "Realistic": "R",
        "Investigative": "I",
        "Artistic": "A",
        "Social": "S",
        "Enterprising": "E",
        "Conventional": "C",
    }

    try:
        code = "".join(mapping[str(x).strip()] for x in labels)
        return code if len(code) == 3 else None
    except Exception:
        return None





# =============================================================================
# Academic helpers
# =============================================================================

def infer_intended_level_from_academic_background(academic_background: Optional[str]) -> Optional[str]:
    if not academic_background:
        return None

    s = academic_background.strip().lower()

    if any(k in s for k in [
        "high school", "secondary school", "diploma", "liceo", "istituto",
        "scientifico", "classico", "linguistico", "professionale", "tecnico", "artistico"
    ]):
        return "bachelor"

    if any(k in s for k in [
        "bachelor", "laurea triennale", "triennale", "undergraduate degree"
    ]):
        return "master"

    if any(k in s for k in [
        "master", "laurea magistrale", "msc", "m.sc", "graduate degree"
    ]):
        return None

    return None

def _infer_program_level(text: str) -> Optional[str]:
    low = (text or "").lower()

    if any(k in low for k in [
        "corso di dottorato",
        "dottorato",
        "phd",
        "doctorate",
        "doctoral",
    ]):
        return "doctorate"

    if any(k in low for k in [
        "laurea magistrale",
        "corso di laurea magistrale",
        "master degree",
        "master's degree",
        "msc",
        "m.sc",
        "m.a",
    ]):
        return "master"

    if any(k in low for k in [
        "laurea triennale",
        "corso di laurea",
        "laurea ",
        "bachelor degree",
        "bachelor's degree",
        "triennale",
    ]):
        return "bachelor"

    return None


def _filter_options_by_intended_level(
    options: List[str],
    intended_level: Optional[str],
) -> List[str]:
    if not intended_level:
        return options

    out: List[str] = []
    for opt in options:
        level = _infer_program_level(opt)
        if level is None or level == intended_level:
            out.append(opt)

    return out

# =============================================================================
# Retrieval block helpers
# =============================================================================
def _is_structured_program_block(text: str) -> bool:
    t = text or ""
    return bool(
        re.search(r"^\s*UNIVERSITÀ:\s*.+$", t, re.MULTILINE)
        and re.search(r"^\s*CORSO:\s*.+$", t, re.MULTILINE)
    )


def _looks_like_catalog_list(text: str) -> bool:
    t = text or ""
    if _is_structured_program_block(t):
        return False

    code_hits = len(re.findall(r"\[[A-Z0-9]{3,10}\]", t))
    english_hits = len(re.findall(r"\bInglese\b", t, re.I))
    course_word_hits = len(re.findall(r"\b(?:CORSO|Percorso|Dottorato|Laurea)\b", t, re.I))
    return code_hits >= 6 or english_hits >= 5 or course_word_hits >= 10



def _advice_final_line() -> str:
    return "Scegli un'opzione indicando il numero." if _DIALOG_LANGUAGE == "it" else "Choose one option by number."


def _extract_numbered_advice_lines(answer: str) -> List[Tuple[int, str]]:
    """Extract numbered advice even when the model prints all options on one line."""
    text = (answer or "").strip()
    if not text:
        return []

    out: List[Tuple[int, str]] = []
    pattern = re.compile(
        r"(?:^|\s)([1-3])\.\s+(.+?)(?=(?:\s+[1-3]\.\s+)|(?:\s+Choose one option by number\.?$)|(?:\s+Scegli un'opzione indicando il numero\.?$)|$)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(text):
        item = re.sub(r"\s+", " ", m.group(2)).strip()
        item = re.sub(r"\s*(?:Choose one option by number|Scegli un'opzione indicando il numero)\.?\s*$", "", item, flags=re.I).strip()
        if item:
            out.append((int(m.group(1)), item))
    return out


def _extract_advice_options_from_answer(answer: str, allowed_options: Sequence[str]) -> List[str]:
    """Return allowed options in the exact order shown by the counselor output."""
    if not answer or not allowed_options:
        return []

    answer_norm = _normalize(answer)
    matches: List[Tuple[int, int, str]] = []
    seen = set()
    for allowed_idx, opt in enumerate(allowed_options):
        if not isinstance(opt, str) or not opt.strip():
            continue
        key = _normalize(opt)
        if not key or key in seen:
            continue
        pos = answer_norm.find(key)
        if pos >= 0:
            seen.add(key)
            matches.append((pos, allowed_idx, opt.strip()))

    if matches:
        matches.sort(key=lambda x: (x[0], x[1]))
        out: List[str] = []
        seen_out = set()
        for _, _, opt in matches:
            k = _normalize(opt)
            if k not in seen_out:
                seen_out.add(k)
                out.append(opt)
            if len(out) >= 3:
                break
        return out

    # Fallback to numbered extraction only if the numbered text exactly matches allowed options.
    allowed_map = {_normalize(opt): opt for opt in allowed_options if isinstance(opt, str) and opt.strip()}
    out = []
    for _, item in _extract_numbered_advice_lines(answer):
        key = _normalize(item)
        if key not in allowed_map:
            return []
        canon = allowed_map[key]
        if _normalize(canon) not in {_normalize(x) for x in out}:
            out.append(canon)
        if len(out) >= 3:
            break
    return out


def _set_last_advice_options_shown(answer: str, allowed_options: Sequence[str]) -> None:
    shown = _extract_advice_options_from_answer(answer, allowed_options)
    if shown:
        global _ADVICE_OPTIONS_SHOWN
        _ADVICE_OPTIONS_SHOWN = list(shown)
        _LAST_RAG_STATE.advice_options = shown
        for opt in shown:
            nopt = _normalize(opt)
            if nopt:
                _LAST_RAG_STATE.allowed_entities.add(nopt)
            if "|" in opt:
                uni = opt.split("|", 1)[0].strip()
                nuni = _normalize(uni)
                if nuni:
                    _LAST_RAG_STATE.allowed_entities.add(nuni)


def _canonicalize_selected_option_value(value: str, shown_options: Optional[Sequence[str]] = None) -> str:
    raw = (value or "").strip()
    options = [str(x).strip() for x in (shown_options or _LAST_RAG_STATE.advice_options or _ADVICE_OPTIONS_SHOWN or []) if str(x).strip()]
    if not raw or not options:
        return raw

    m = re.match(r"^\s*([1-3])\s*(?:[-.)]|$)", raw)
    if m:
        n = int(m.group(1))
        if 1 <= n <= len(options):
            return f"{n} - {options[n - 1]}"

    raw_norm = _normalize(raw)
    for i, opt in enumerate(options, start=1):
        opt_norm = _normalize(opt)
        if opt_norm and (opt_norm in raw_norm or raw_norm.endswith(opt_norm)):
            return f"{i} - {opt}"

    return raw


def _validate_advice_answer(answer: str, allowed_options: Sequence[str]) -> Optional[str]:
    chosen = _extract_advice_options_from_answer(answer, allowed_options)
    if not chosen:
        return None

    final_line = _advice_final_line()
    return "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(chosen)] + [final_line])


def _course_domain_mismatch(course: str, field_of_interest: Any) -> bool:
    c = _normalize(course)

    if not c or not field_of_interest:
        return False

    if isinstance(field_of_interest, str):
        fields = [_normalize(field_of_interest)]
    elif isinstance(field_of_interest, (list, tuple)):
        fields = [_normalize(x) for x in field_of_interest if isinstance(x, str) and x.strip()]
    else:
        fields = []

    if not fields:
        return False

    positive = {
        "arts_design": [
            "design", "arte", "art", "arts", "arti", "artistic", "artistico", "artistica",
            "visual", "grafica", "creative", "creativo", "creativa",
            "architettura", "architecture", "progetto", "progettazione",
        ],
        "communication_digital_media": [
            "communication", "comunicazione", "media", "digital", "digitale",
            "visual communication", "communication design", "media production", "data visualization",
            "grafica", "design",
        ],
        "humanities": [
            "lettere", "literature", "literary", "philology", "filologia",
            "philosophy", "filosofia", "history", "historical", "storia",
            "languages", "lingue", "language", "cultural", "culture", "cultura",
            "beni culturali", "humanities", "umanistiche",
        ],
        "cultural_heritage": [
            "beni culturali", "cultural heritage", "heritage", "patrimonio culturale",
            "art history", "storia dell arte", "archeology", "archaeology",
            "archeologia", "museum", "musei", "cultural", "culture",
        ],
        "social_sciences": [
            "sociology", "sociologia", "social", "political", "politics",
            "international studies", "global", "local studies", "education", "pedagogy",
        ],
        "environmental_science": [
            "environment", "environmental", "ambiente", "ambientale",
            "ecology", "ecologia", "ecosystem", "ecosystems", "biodiversity",
            "conservation", "territorio", "paesaggio", "sustainability", "sostenibil",
            "natural sciences", "scienze naturali", "ecotoxicology",
        ],
        "computer_science": [
            "computer science", "informatica", "software", "data", "data science",
            "artificial intelligence", "intelligenza artificiale", "cybersecurity",
            "information systems", "computer engineering",
        ],
        "health_technology": [
            "health", "healthcare", "sanitar", "biomedical", "biomed",
            "neurophysiopathology", "neurofisiopatologia", "rehabilitation",
            "riabilitazione", "medical technology",
        ],
        "life_sciences": [
            "biology", "biologia", "biological", "biologico", "biologica",
            "biotechnology", "biotechnologies", "biotecnologie", "biomolecular",
            "molecular", "cellular", "life sciences", "scienze della vita",
            "conservation biology",
        ],
        "engineering_technology": [
            "engineering", "ingegneria", "mechanical", "meccanica", "civil", "civile",
            "electronic", "electronics", "elettronica", "mechatronics", "meccatronica",
            "industrial", "industriale", "computer engineering", "technical systems",
            "technology", "technologies", "tecnologia", "tecnologie",
        ],
        "physical_sciences": [
            "physics", "fisica", "physical", "chemistry", "chimica",
            "mathematics", "matematica", "applied sciences", "scienze applicate",
            "planetary sciences", "scienze planetarie", "astronomy", "astronomia",
        ],
        "environmental_engineering": [
            "environmental engineering", "ingegneria ambientale", "engineering for the environment",
            "ingegneria per l ambiente", "territory", "territorio", "ambiente e territorio",
            "environmental technology", "tecnologia ambientale", "technical conservation systems",
        ],
        "psychology": [
            "psychology", "psicologia", "clinical psychology", "psicologia clinica",
            "counseling", "counselling", "psychotherapy", "psicoterapia",
            "mental health", "salute mentale", "behavior", "behaviour", "comportamento",
        ],
        "cognitive_science_linguistics": [
            "cognitive science", "scienze cognitive", "linguistics", "linguistica",
            "mind", "mente", "language", "linguaggio", "neuroscience", "neuroscienze",
        ],
    }

    hard_negative = {
        "arts_design": [
            "medicine", "medicina", "nursing", "infermieristica", "pharmacy", "farmacia",
            "odontoiatria", "chemistry", "chimica", "physics", "fisica", "economics",
            "finance", "statistica", "law", "giurisprudenza", "engineering", "ingegneria",
            "electronic", "computer science", "biotechnology",
        ],
        "communication_digital_media": [
            "medicine", "pharmacy", "biotechnology", "civil engineering", "electronic engineering",
            "mechanical engineering",
        ],
        "humanities": [
            "engineering", "ingegneria", "computer science", "informatica", "medicine",
            "medicina", "pharmacy", "farmacia", "biotechnology", "biotecnologie",
        ],
        "cultural_heritage": [
            "medicine", "medicina", "pharmacy", "farmacia", "computer science",
            "informatica pura", "biotechnology", "biotecnologie",
        ],
        "environmental_science": [
            "odontoiatria", "pharmacy", "farmacia", "law", "giurisprudenza",
            "economics", "economia pura",
        ],
        "engineering_technology": [
            "law", "giurisprudenza", "philology", "literature", "psychology",
            "education", "foreign languages", "biotechnology", "biotechnologies",
            "biology", "biological", "medicine", "pharmacy",
        ],
        "computer_science": [
            "law", "giurisprudenza", "philology", "literature", "psychology",
            "education", "foreign languages", "biotechnology", "biology",
        ],
    }

    fields_set = set(fields)

    def has_any(kws: Sequence[str]) -> bool:
        return _contains_any_keyword(course, kws)

    # Cross-domain guards. These avoid false positives such as
    # technology -> biotechnology and art -> artificial intelligence.
    if "engineering_technology" in fields_set and not ({"life_sciences", "health_technology"} & fields_set):
        life_markers = ["biotechnology", "biotechnologies", "biology", "biological", "biomolecular", "medicine", "pharmacy"]
        engineering_markers = ["engineering", "ingegneria", "electronic", "electronics", "mechanical", "civil", "industrial", "mechatronics", "computer engineering"]
        if has_any(life_markers) and not has_any(engineering_markers):
            return True

    if "computer_science" in fields_set and "life_sciences" not in fields_set:
        if has_any(["biotechnology", "biotechnologies", "biology", "biological", "biomolecular"]) and not has_any(["computer", "data", "software", "artificial intelligence", "informatics"]):
            return True

    # Hard negative: if the course is clearly out of domain for any primary field, discard.
    for field in fields:
        for kw in hard_negative.get(field, []):
            if _keyword_in_normalized_text(course, kw):
                return True

    # Positive match: at least one inferred field must have a real token/phrase match.
    for field in fields:
        kws = positive.get(field, [])
        if any(_keyword_in_normalized_text(course, kw) for kw in kws):
            return False

    strict_fields = {
        "arts_design", "communication_digital_media", "humanities", "cultural_heritage",
        "social_sciences", "environmental_science", "environmental_engineering",
        "life_sciences", "computer_science", "engineering_technology", "physical_sciences",
        "health_technology", "psychology", "cognitive_science_linguistics",
    }
    if any(field in strict_fields for field in fields):
        return True

    return False

def _extract_ctx_options(
    ctx: str,
    max_items: int = 3,
    intended_level: Optional[str] = None,
    field_of_interest: Any = None,
    debug: bool = False,
) -> List[str]:
    raw = (ctx or "").strip()
    if not raw:
        return []

    blocks = [b.strip() for b in re.split(r"\n\s*---\s*\n", raw) if b.strip()]
    options: List[str] = []
    seen = set()

    for block in blocks:
        university, course, tipo = _parse_program_block(block)

        if not university or not course:
            _dprint(debug, f"[DEBUG][extract] reject=parse university={university!r} course={course!r}")
            continue

        disable_domain_filter = os.getenv("SDIALOG_DISABLE_DOMAIN_FILTER", "0").lower() in {"1", "true", "yes"}
        if (not disable_domain_filter) and _course_domain_mismatch(course, field_of_interest):
            _dprint(debug, f"[DEBUG][extract] reject=domain course={course}")
            continue

        level = _infer_program_level(f"{tipo}\n{block}")

        if level == "doctorate" and intended_level != "doctorate":
            _dprint(debug, f"[DEBUG][extract] reject=doctorate course={course} level={level} intended={intended_level}")
            continue

        if intended_level:
            if level is not None and level != intended_level:
                _dprint(debug, f"[DEBUG][extract] reject=level course={course} level={level} intended={intended_level}")
                continue

        option = f"{university} | {course}"
        key = option.casefold()
        if key not in seen:
            seen.add(key)
            options.append(option)
            _dprint(debug, f"[DEBUG][extract] accept option={option}")

        if len(options) >= max_items:
            break

    if options:
        return options

    if intended_level:
        return []

    for block in blocks:
        university, _, _ = _parse_program_block(block)
        if not university:
            continue
        key = university.casefold()
        if key not in seen:
            seen.add(key)
            options.append(university)
        if len(options) >= max_items:
            break

    return options




def _extract_allowed_entities_from_ctx(ctx: str) -> set[str]:
    out: set[str] = set()
    for m in _ENTITY_RX.finditer(ctx or ""):
        out.add(_normalize(m.group(0)))
    for o in _extract_ctx_options(ctx, max_items=8):
        out.add(_normalize(o))
    return {x for x in out if x}

def _hit_to_program_block(hit: Dict[str, Any]) -> str:
        meta = hit.get("meta") or {}
        text = (hit.get("text") or "").strip()

        university = (
                str(meta.get("UNIVERSITY") or "").strip()
                or str(meta.get("UNIVERSITÀ") or "").strip()
                or str(meta.get("UNIVERSITA") or "").strip()
                or str(meta.get("university") or "").strip()
        )

        course = (
                str(meta.get("COURSE") or "").strip()
                or str(meta.get("CORSO") or "").strip()
                or str(meta.get("course") or "").strip()
        )

        code = str(meta.get("COURSE_CODE") or "").strip()
        tipo = (
                str(meta.get("DEGREE_TYPE") or "").strip()
                or str(meta.get("TIPO") or "").strip()
                or str(meta.get("TYPE") or "").strip()
        )

        if code and course and not course.startswith("["):
            course = f"[{code}] {course}"

        lines = []
        if university:
            lines.append(f"UNIVERSITÀ: {university}")
        if course:
            lines.append(f"CORSO: {course}")
        if tipo:
            lines.append(f"TIPO: {tipo}")
        if text:
            lines.append(text)

        return "\n".join(lines).strip()

def _select_option_context_blocks(raw_blocks: List[str], options: List[str], max_blocks: int = 4) -> List[str]:
    if not raw_blocks:
        return []

    wanted = {_option_key(x) for x in options if x}
    if not wanted:
        return raw_blocks[:max_blocks]

    matched: List[str] = []
    for block in raw_blocks:
        university, course, _ = _parse_program_block(block)
        if not university or not course:
            continue
        option = f"{university} | {course}"
        if _option_key(option) in wanted:
            matched.append(block)
        if len(matched) >= max_blocks:
            break

    return matched or raw_blocks[:max_blocks]


def _build_short_ctx(
    raw_blocks: List[str],
    max_chars: int,
    max_blocks: int = 8,
    min_block_chars: int = 220,
) -> str:
    if not raw_blocks or max_chars <= 0:
        return ""

    sep = "\n\n---\n\n"
    sep_cost = len(sep)

    blocks = [(b or "").strip() for b in raw_blocks if (b or "").strip()]
    if not blocks:
        return ""

    blocks = blocks[:max_blocks]

    out: List[str] = []
    remaining = max_chars

    for idx, block in enumerate(blocks):
        blocks_left = len(blocks) - idx
        reserve_for_separators = sep_cost * max(0, blocks_left - 1)

        if remaining <= reserve_for_separators + 40:
            break

        available_now = remaining - reserve_for_separators

        # quota dinamica per non far divorare tutto ai primi blocchi
        fair_share = max(min_block_chars, available_now // blocks_left)
        take = min(len(block), fair_share)

        piece = block[:take].rstrip()
        if not piece:
            continue

        extra_cost = sep_cost if out else 0
        if len(piece) + extra_cost > remaining:
            piece = piece[: max(0, remaining - extra_cost)].rstrip()

        if not piece:
            break

        if out:
            out.append(sep.strip())
            remaining -= sep_cost

        out.append(piece)
        remaining -= len(piece)

        if remaining <= 40:
            break

    return "\n\n".join(out).strip()

# =============================================================================
# Canonical listener memory
# =============================================================================
def canonical_listener_memory() -> Dict[str, Any]:
    return {
        "explicit": {
            "academic_background": None,
            "region": None,
            "selected_option": None,
        },
        "inferred": {
            "gender": None,
            "field_of_interest": None,
            "riasec_attitudes": None,
            "riasec_confidence": None,
        },
    }


def _empty_listener_memory() -> dict:
    return canonical_listener_memory()

def _normalize_field_of_interest(value: Any) -> Optional[List[str]]:
    allowed = set(FIELD_OF_INTEREST_LABELS)

    if isinstance(value, str):
        s = value.strip()
        if s in allowed:
            return [s]
        return None

    if not isinstance(value, (list, tuple)):
        return None

    out = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s in allowed and s not in seen:
            seen.add(s)
            out.append(s)

    return out[:3] if out else None

def apply_listener_patch(mem: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(mem, dict):
        mem = _empty_listener_memory()
    if not isinstance(patch, dict):
        return mem

    exp = mem.get("explicit", {})
    inf = mem.get("inferred", {})
    if not isinstance(exp, dict):
        exp = {}
    if not isinstance(inf, dict):
        inf = {}

    pexp = patch.get("explicit", {})
    pinf = patch.get("inferred", {})
    if not isinstance(pexp, dict):
        pexp = {}
    if not isinstance(pinf, dict):
        pinf = {}

    def keep_more_info(old: Optional[str], new: Optional[str]) -> Optional[str]:
        old = (old or "").strip()
        new = (new or "").strip()
        if not new:
            return old or None
        if not old:
            return new
        if new.casefold() == old.casefold():
            return old
        return new if len(new) >= len(old) else old

    region_value = _normalize_region_value(pexp.get("region"))
    if region_value:
        exp["region"] = region_value

    if isinstance(pexp.get("academic_background"), str) and pexp["academic_background"].strip():
        exp["academic_background"] = keep_more_info(
            exp.get("academic_background"),
            pexp["academic_background"],
        )

    if isinstance(pexp.get("selected_option"), str) and pexp["selected_option"].strip():
        # The listener extracts the student's choice via prompt. Python only
        # canonicalizes that prompt-extracted value against the options that were
        # actually shown to the student, avoiding stale candidate-pool mappings.
        exp["selected_option"] = _canonicalize_selected_option_value(pexp["selected_option"].strip())

    foi = _normalize_field_of_interest(pinf.get("field_of_interest"))
    if foi:
        inf["field_of_interest"] = foi

    if isinstance(pinf.get("gender"), str) and pinf["gender"].strip():
        inf["gender"] = pinf["gender"].strip()

    valences = _normalize_riasec_valences(pinf.get("riasec_question_valences"))
    if valences:
        top3, confidence = _riasec_top3_and_confidence_from_valences(valences)
        inf["riasec_attitudes"] = top3
        inf["riasec_confidence"] = confidence
    else:
        labels = _normalize_riasec_labels(pinf.get("riasec_attitudes"))
        if labels:
            inf["riasec_attitudes"] = labels
            inf["riasec_confidence"] = inf.get("riasec_confidence") or "unknown"

    mem["explicit"] = exp
    mem["inferred"] = inf
    return mem



# =============================================================================
# Debug snapshots
# =============================================================================
class DebugSnapshot(TypedDict):
    listener_memory: dict
    slots_summary: str


_DEBUG_SNAPSHOTS: List[DebugSnapshot] = []


def clear_debug_snapshots() -> None:
    _DEBUG_SNAPSHOTS.clear()


def record_debug_snapshot(listener_memory: dict, slots_summary: str) -> None:
    _DEBUG_SNAPSHOTS.append(
        {
            "listener_memory": copy.deepcopy(listener_memory),
            "slots_summary": str(slots_summary or "").strip(),
        }
    )


def get_debug_snapshots() -> List[DebugSnapshot]:
    return list(_DEBUG_SNAPSHOTS)


# =============================================================================
# Listener patch buffer
# =============================================================================
_LISTENER_PATCH_HISTORY: List[Dict[str, Any]] = []
_LISTENER_PATCH_BUFFER: List[Dict[str, Any]] = []
_ADVICE_OPTIONS_SHOWN: List[str] = []


def clear_listener_patches() -> None:
    _LISTENER_PATCH_HISTORY.clear()
    _LISTENER_PATCH_BUFFER.clear()


def get_listener_patches() -> List[Dict[str, Any]]:
    return list(_LISTENER_PATCH_HISTORY)




def _canonicalize_listener_patch_with_shown_options(data: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize prompt-extracted selected_option before it reaches logs/memory.

    The listener still extracts the information by prompt. Python only normalizes
    that prompt-extracted value against the options actually displayed to the
    student, so a stale candidate-pool option cannot leak into the logged memory.
    """
    if not isinstance(data, dict):
        return data

    exp = data.get("explicit")
    if not isinstance(exp, dict):
        return data

    selected = exp.get("selected_option")
    if not isinstance(selected, str) or not selected.strip():
        return data

    try:
        selected = _canonicalize_selected_option_value(selected)
    except Exception:
        return data

    out = copy.deepcopy(data)
    out.setdefault("explicit", {})["selected_option"] = selected
    return out


def strip_and_buffer_listener_patch(text: str) -> str:
    t = text or ""
    while True:
        m = _LISTENER_PATCH_BLOCK_RX.search(t)
        if not m:
            break
        raw_json = m.group(1)
        data = _safe_json_loads(raw_json)
        if isinstance(data, dict):
            data = _canonicalize_listener_patch_with_shown_options(data)
            _LISTENER_PATCH_BUFFER.append(data)
            _LISTENER_PATCH_HISTORY.append(copy.deepcopy(data))
        t = (t[: m.start()] + t[m.end() :]).strip()
    return t


def _pop_listener_patch() -> Optional[Dict[str, Any]]:
    if not _LISTENER_PATCH_BUFFER:
        return None
    return _LISTENER_PATCH_BUFFER.pop(0)


def _extract_listener_patch_block(text: str) -> str:
    m = _LISTENER_PATCH_BLOCK_RX.search(text or "")
    return m.group(0).strip() if m else ""


# =============================================================================
# RAG state
# =============================================================================
@dataclass
class _RAGState:
    ctx: str = ""
    query: str = ""
    phase: str = ""
    expected_verbatim: str = ""
    advice_fallback: bool = False
    allowed_entities: set[str] = field(default_factory=set)
    advice_options: List[str] = field(default_factory=list)
    choice_options: List[str] = field(default_factory=list)


_LAST_RAG_STATE = _RAGState()


def update_last_rag_state(
    ctx: str,
    *,
    query: str = "",
    phase: str = "",
    expected_verbatim: str = "",
    advice_fallback: bool = False,
    advice_options: Optional[Sequence[str]] = None,
    choice_options: Optional[Sequence[str]] = None,
) -> None:
    _LAST_RAG_STATE.ctx = ctx or ""
    _LAST_RAG_STATE.query = query or ""
    _LAST_RAG_STATE.phase = (phase or "").strip().lower()
    _LAST_RAG_STATE.expected_verbatim = (expected_verbatim or "").strip()
    _LAST_RAG_STATE.advice_fallback = bool(advice_fallback)

    if advice_options is None:
        # Preserve the ranked/displayed advice list across Q&A and wrap-up phases.
        # Otherwise a later state update with advice_options=None clears the list
        # needed to resolve prompt-extracted choices like "option 1".
        advice_source = list(_LAST_RAG_STATE.advice_options or _ADVICE_OPTIONS_SHOWN or [])
    else:
        advice_source = list(advice_options or [])

    _LAST_RAG_STATE.advice_options = [
        str(x).strip()
        for x in advice_source
        if str(x).strip()
    ]
    _LAST_RAG_STATE.choice_options = [
        str(x).strip()
        for x in (choice_options or [])
        if str(x).strip()
    ]
    _LAST_RAG_STATE.allowed_entities = _extract_allowed_entities_from_ctx(_LAST_RAG_STATE.ctx)

    for opt in _LAST_RAG_STATE.advice_options:
        nopt = _normalize(opt)
        if nopt:
            _LAST_RAG_STATE.allowed_entities.add(nopt)
        if "|" in opt:
            uni = opt.split("|", 1)[0].strip()
            nuni = _normalize(uni)
            if nuni:
                _LAST_RAG_STATE.allowed_entities.add(nuni)


def clear_last_rag_state() -> None:
    global _ADVICE_OPTIONS_SHOWN
    _ADVICE_OPTIONS_SHOWN = []
    update_last_rag_state(
        "",
        query="",
        phase="",
        expected_verbatim="",
        advice_fallback=False,
        advice_options=[],
        choice_options=[],
    )


def get_last_rag_phase() -> str:
    return (_LAST_RAG_STATE.phase or "").strip().lower()


def _normalize_wrapup_answer(ans: str) -> str:
    body = _LISTENER_PATCH_BLOCK_RX.sub("", ans or "").strip()

    recap = ""
    code = "UNK"
    goodbye = "Best of luck with your studies!" if _DIALOG_LANGUAGE != "it" else "In bocca al lupo per il tuo percorso!"

    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("recap:"):
            recap = s.split(":", 1)[1].strip()
        elif s.lower().startswith("riasec:"):
            maybe = s.split(":", 1)[1].strip().split()[0].strip(" .;,")
            if maybe:
                code = maybe
        elif s.lower().startswith("goodbye:"):
            maybe = s.split(":", 1)[1].strip()
            if maybe:
                goodbye = maybe

    if not recap:
        recap = (
            "You chose a grounded option based on your background, preferences, constraints, and RIASEC profile."
            if _DIALOG_LANGUAGE != "it"
            else "Hai scelto un'opzione grounded in base a background, preferenze, vincoli e profilo RIASEC."
        )

    if _DIALOG_LANGUAGE == "it":
        steps = [
            "1. rivedere il piano ufficiale del corso e i requisiti di accesso",
            "2. verificare costi, sede, pendolarismo o vincoli abitativi",
            "3. preparare documenti, portfolio o revisione dei prerequisiti",
        ]
    else:
        steps = [
            "1. review the official course plan and admission requirements",
            "2. verify costs, location, commuting, or housing constraints",
            "3. prepare documents, portfolio, or prerequisite review",
        ]

    return "\n".join([
        f"Recap: {recap}",
        f"RIASEC: {code}",
        "Next steps:",
        *steps,
        f"Goodbye: {goodbye}",
    ])


def _done_no_options_message() -> str:
    if _DIALOG_LANGUAGE == "it":
        return (
            "Non ho trovato opzioni grounded sufficientemente adatte nell'insieme universitario corrente. "
            "Prossimi passi suggeriti: allarga leggermente il campo target; mantieni il campo esatto e ripeti la ricerca quando saranno disponibili più università; rivedi vincoli e priorità. "
            "Arrivederci."
        )
    return (
        "I could not find grounded options that fit closely enough in the current university set. "
        "Suggested next steps: broaden the target field slightly; keep the field exact and search again when more universities are available; review constraints and priorities. "
        "Goodbye."
    )


def enforce_grounding_or_fallback(answer: str) -> str:
    ans = (answer or "").strip()
    if not ans:
        return ans

    phase = (_LAST_RAG_STATE.phase or "").strip().lower()
    ctx = _LAST_RAG_STATE.ctx or ""
    ctx_n = _normalize(ctx)

    if phase not in {"advice", "qa_why", "qa_other", "qa_yesno", "qa_choice", "qa_selected", "wrapup"}:
        return ans
    if phase == "advice":
        # In advice, let the dedicated advice validator/formatter handle structure.
        # Do not replace the model ranking with a generic fallback here.
        return ans

    if phase == "wrapup":
        # The wrap-up contains generic checklist items. Do not run admission/cost
        # grounding filters here, otherwise items 1 and 2 are deleted.
        return ans

    if not ctx.strip():
        return ans

    if phase == "advice" and _LAST_RAG_STATE.advice_fallback:
        return ans

    kept: List[str] = []
    removed_admission_line = False

    for line in ans.splitlines():
        s = line.strip()
        if not s:
            kept.append(line)
            continue

        drop = False
        for rx, keys in _REQ_RXES:
            if rx.search(s) and not _ctx_has_any(ctx_n, keys):
                drop = True
                if any(k in " ".join(keys).lower() for k in ["deadline", "apply", "admission", "enroll"]):
                    removed_admission_line = True
                break

        if not drop:
            kept.append(line)

    ans = "\n".join(kept).strip() or ans

    if removed_admission_line and phase in {"advice", "qa_other"}:
        if "verify on official channels" not in ans.lower():
            ans = (
                ans
                + "\n\nAdmissions details can vary by program; please verify on official channels."
            ).strip()

    ents = [m.group(0).strip() for m in _ENTITY_RX.finditer(ans)]
    ungrounded = []
    for e in ents:
        en = _normalize(e)
        if en in ctx_n:
            continue
        if any(en in a or a in en for a in _LAST_RAG_STATE.allowed_entities):
            continue
        ungrounded.append(e)

    if not ungrounded:
        return ans

    if phase in {"qa_why", "wrapup"}:
        def redact_ent(m: re.Match) -> str:
            tok = m.group(0)
            tn = _normalize(tok)
            if tn in ctx_n or any(tn in a or a in tn for a in _LAST_RAG_STATE.allowed_entities):
                return tok
            return "the selected program/university"

        return _ENTITY_RX.sub(redact_ent, ans)

    options = list(_LAST_RAG_STATE.advice_options or [])
    if not options:
        options = _extract_ctx_options(ctx, max_items=3)

    if options:
        opt_lines = "\n".join([f"{i + 1}. {o}" for i, o in enumerate(options)])
        return (
            "Here are the grounded options I can support:\n"
            f"{opt_lines}\n"
            f"Which option do you want to pursue (1-{len(options)})?"
        ).strip()

    return ans


def enforce_flow_format_or_fallback(answer: str) -> str:
    ans = (answer or "").strip()
    phase = (_LAST_RAG_STATE.phase or "").strip().lower()
    expected = (_LAST_RAG_STATE.expected_verbatim or "").strip()
    ctx = _LAST_RAG_STATE.ctx or ""
    advice_fallback = bool(getattr(_LAST_RAG_STATE, "advice_fallback", False))

    yes_tok, no_tok = ("Sì", "No") if _DIALOG_LANGUAGE == "it" else ("Yes", "No")
    patch_block = _extract_listener_patch_block(ans)

    if phase == "qa_yesno":
        body = _LISTENER_PATCH_BLOCK_RX.sub("", ans).strip()
        out = None
        if re.fullmatch(r"(?i)\s*(yes|sì|si)\s*", body):
            out = yes_tok
        elif re.fullmatch(r"(?i)\s*no\s*", body):
            out = no_tok
        else:
            cleaned = re.sub(r"[\.!\?\"'\,]+", "", body).strip()
            if re.fullmatch(r"(?i)\s*(yes|sì|si)\s*", cleaned):
                out = yes_tok
            elif re.fullmatch(r"(?i)\s*no\s*", cleaned):
                out = no_tok
            else:
                yes = bool(_FLOW_YES_RX.search(body))
                no = bool(_FLOW_NO_RX.search(body))
                if yes and not no:
                    out = yes_tok
                elif no and not yes:
                    out = no_tok

        # Do not convert ambiguous/non-compliant answers to a synthetic "No".
        # Leave them visible so the run can be marked invalid instead of biased.
        if out is None:
            out = body or ans
        return (out + ("\n" + patch_block if patch_block else "")).strip()

    if phase == "qa_choice":
        body = _LISTENER_PATCH_BLOCK_RX.sub("", ans).strip()
        allowed = list(_LAST_RAG_STATE.choice_options or [])
        if allowed:
            norm_allowed = {_normalize(x): x for x in allowed}
            key = _normalize(body)
            if key in norm_allowed:
                return (norm_allowed[key] + ("\n" + patch_block if patch_block else "")).strip()

            body_norm = _normalize(body)
            hits = [x for x in allowed if _normalize(x) and _normalize(x) in body_norm]
            if len(hits) == 1:
                return (hits[0] + ("\n" + patch_block if patch_block else "")).strip()

            # A bare Yes/No is not one of the two requested alternatives.
            # Preserve it as invalid/non-compliant instead of silently choosing.
            return (body + ("\n" + patch_block if patch_block else "")).strip()

    if expected and phase.startswith("riasec_"):
        out = expected
        return (out + ("\n" + patch_block if patch_block else "")).strip()

    if phase == "greet":
        out = (
            _t(
                "Hello! I help students choose university programs based on their background, interests, aptitudes, and constraints.",
                "Ciao! Aiuto gli studenti a scegliere un percorso universitario in base al loro background, ai loro interessi, alle attitudini e ai vincoli."
            )
            + "\n"
            + expected
        ).strip()
        if patch_block:
            out = f"{out}\n{patch_block}"
        return out

    if expected and phase.startswith("background_"):
        out = f"{_t('Thanks, that helps.', 'Grazie, è molto utile.')}\n{expected}".strip()
        if patch_block:
            out = f"{out}\n{patch_block}"
        return out

    if phase == "advice" and advice_fallback:
        out = expected or _t(
            "Could you clarify one point before I recommend programs: what matters most to you right now—degree level or exact field?",
            "Potresti chiarire un punto prima che ti consigli dei corsi: che cosa conta di più per te in questo momento tra livello del titolo o ambito preciso?"
        )
        if patch_block:
            out = f"{out}\n{patch_block}"
        return out

    if phase == "advice":
        allowed = list(_LAST_RAG_STATE.advice_options or [])
        validated = _validate_advice_answer(ans, allowed)

        if validated:
            out = validated
        else:
            fallback = allowed[: min(3, len(allowed))]
            if fallback:
                out = "\n".join(
                    [f"{i + 1}. {opt}" for i, opt in enumerate(fallback)] + [_advice_final_line()]
                )
            else:
                out = ans
        _set_last_advice_options_shown(out, allowed)
        if patch_block:
            out = f"{out}\n{patch_block}"
        return out

    if phase == "wrapup":
        out = _normalize_wrapup_answer(ans)
        if patch_block:
            out = f"{out}\n{patch_block}"
        return out

    if phase == "clarify_scope":
        out = expected or _t(
            "I could not find grounded options yet. Could you clarify one point so I can narrow the search better?",
            "Non ho ancora trovato opzioni grounded. Potresti chiarire un punto così posso restringere meglio la ricerca?"
        )
        if patch_block:
            out = f"{out}\n{patch_block}"
        return out
    if phase == "done":
        out = _done_no_options_message()
        if patch_block:
            out = f"{out}\n{patch_block}"
        return out

    if phase == "stop":
        return "STOP"

    return ans

# =============================================================================
# Student slots
# =============================================================================
@dataclass
class StudentSlots:
    academic_background: Optional[str] = None
    region: Optional[str] = None
    field_of_interest: Optional[str] = None
    intended_level: Optional[str] = None
    selected_option: Optional[str] = None

    def summary(self) -> str:
        parts = []
        if self.academic_background:
            parts.append(f"academic_background={self.academic_background}")
        if self.field_of_interest:
            parts.append(f"field_of_interest={self.field_of_interest}")
        if self.region:
            parts.append(f"region={self.region}")
        if self.intended_level:
            parts.append(f"intended_level={self.intended_level}")
        if self.selected_option:
            parts.append(f"selected_option={self.selected_option}")
        return "; ".join(parts)


# =============================================================================
# Retrieval result
# =============================================================================
@dataclass
class RetrievalScopeResult:
    scope_name: str = "none"
    raw_blocks: List[str] = field(default_factory=list)
    short_ctx: str = ""
    options: List[str] = field(default_factory=list)
    preferred_zone: Optional[str] = None
    matched_zone: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.raw_blocks and not self.options

    @property
    def expanded(self) -> bool:
        return bool(
            self.preferred_zone
            and self.matched_zone
            and self.preferred_zone != self.matched_zone
        )


# =============================================================================
# Orchestrator
# =============================================================================
class UniversityCounselorFlowOrchestrator(BaseOrchestrator):

    _CHOICE_CONNECTOR = re.compile(r"\s+(?:or|vs\.?|versus)\s+", re.I)
    _WH_RX = re.compile(r"\b(why|how|what|which|where|when)\b", re.IGNORECASE)
    _REQ_RX = re.compile(r"\b(could you|can you|please|clarify|outline|explain|share|tell me)\b", re.IGNORECASE)
    _AUX_YN_RX = re.compile(r"^\s*(is|are|am|do|does|did|can|could|would|should|will|have|has)\b", re.IGNORECASE)

    def __init__(
            self,
            *,
            retriever: Any,
            required_slots: Sequence[str] = ("academic_background", "field_of_interest", "region"),
            top_k: int = 12,
            max_ctx_chars: int = 3200,
            history_turns: int = 3,
            debug: bool = False,
            min_options_target: int = 3,
            candidate_pool_size: int = 8,
            final_top_n: int = 3,
            dialog_language: str = "English",
            practice: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.lang = _lang_code(dialog_language)
        self.practice = practice or default_university_counseling_practice()

        self.agent1_norms = [
            str(x).strip()
            for x in (self.practice.get("agent1_norms") or [])
            if isinstance(x, str) and x.strip()
        ]

        self.BG_QUESTIONS = _require_practice_list(
            self.practice,
            "dialogue_config",
            "counselor",
            "background_questions",
            dialog_language=dialog_language,
            min_len=1,
        )

        self.RIASEC_QUESTIONS = _require_practice_list(
            self.practice,
            "dialogue_config",
            "counselor",
            "riasec_questions",
            dialog_language=dialog_language,
            exact_len=8,
        )

        self.FIXED_FOLLOWUP_QUESTIONS = get_practice_list(
            self.practice,
            "dialogue_config",
            "student",
            "fixed_followup_questions",
            dialog_language=dialog_language,
            default=[],
        )

        self.retriever = retriever
        # required_slots is accepted for backward compatibility with agents_setup.py.
        self.top_k = int(top_k)
        self.max_ctx_chars = int(max_ctx_chars)
        # history_turns is accepted for backward compatibility with agents_setup.py.
        self.debug = bool(debug)
        # min_options_target is accepted for backward compatibility.

        self._slots = StudentSlots()
        self._listener_memory = _empty_listener_memory()

        self._flow_stage = "greet"
        self._bg_i = 0
        self._riasec_i = 0
        self._last_ctx: str = ""
        self._advice_given = False
        self._pending_selection_ack = False
        self._pending_selection_number: Optional[int] = None
        self._selection_patch_requested_for: Optional[int] = None

        self._riasec_patch_requested = False
        self._choice_patch_requested = False

        self._last_options: List[str] = []
        self._riasec_answers: List[str] = []
        self._finished = False
        self._advice_retry_count = 0
        self.candidate_pool_size = int(candidate_pool_size)
        self.final_top_n = int(final_top_n)

        if self.lang == "it":
            self._CHOICE_CONNECTOR = re.compile(r"\s+(?:or|oppure|o|vs\.?|versus)\s+", re.I)
            self._WH_RX = re.compile(r"\b(why|how|what|which|where|when|perche|perché|come|cosa|quale|dove|quando)\b",
                                     re.I)
            self._REQ_RX = re.compile(
                r"\b(could you|can you|please|clarify|outline|explain|share|tell me|puoi|potresti|per favore|chiarire|spiegare|dirmi)\b",
                re.I)
            self._AUX_YN_RX = re.compile(
                r"^\s*(is|are|am|do|does|did|can|could|would|should|will|have|has|pensi|credi|ritieni|consigli|suggerisci|potrei|mi troverei)\b",
                re.I)
        else:
            self._CHOICE_CONNECTOR = re.compile(r"\s+(?:or|vs\.?|versus)\s+", re.I)
            self._WH_RX = re.compile(r"\b(why|how|what|which|where|when)\b", re.I)
            self._REQ_RX = re.compile(r"\b(could you|can you|please|clarify|outline|explain|share|tell me)\b", re.I)
            self._AUX_YN_RX = re.compile(r"^\s*(is|are|am|do|does|did|can|could|would|should|will|have|has)\b", re.I)


    def reset(self) -> None:
        self._slots = StudentSlots()
        self._listener_memory = _empty_listener_memory()

        self._flow_stage = "greet"
        self._bg_i = 0
        self._riasec_i = 0
        self._last_ctx = ""
        self._advice_given = False
        self._pending_selection_ack = False
        self._pending_selection_number: Optional[int] = None
        self._selection_patch_requested_for: Optional[int] = None

        self._riasec_patch_requested = False
        self._choice_patch_requested = False

        clear_last_rag_state()
        clear_listener_patches()
        self._last_options = []
        self._riasec_answers = []
        self._finished = False
        self._advice_retry_count = 0

    # -------------------------------------------------------------------------
    # Listener patch prompt builder
    # -------------------------------------------------------------------------
    def _listener_patch_request(
        self,
        targets: str,
        schema_example: str,
        evidence: str,
        rules: str = "",
    ) -> str:
        return (
            "\nHIDDEN LISTENER TASK (STRICT):\n"
            "- After your student-visible reply, append ONE <LISTENER_PATCH> JSON block.\n"
            "- The patch must include ONLY the requested target fields.\n"
            "- If a field is unknown, omit it (do NOT guess).\n"
            "- Output valid JSON only inside the tags.\n"
            f"TARGET FIELDS:\n{targets}\n"
            + (f"RULES:\n{rules}\n" if rules else "")
            + f"SCHEMA EXAMPLE:\n{schema_example}\n"
              f"EVIDENCE:\n{evidence}\n"
        )

    def _background_slot_patch_request(self, evidence: str) -> str:
        return self._listener_patch_request(
            targets="- explicit.academic_background\n- inferred.field_of_interest\n- explicit.region",
            rules=(
                "Rules for explicit.academic_background:\n"
                "- If the student states any current or completed formal education, you MUST include academic_background.\n"
                "- academic_background must summarize only formal education, not jobs, hobbies, internships, goals, or personal interests.\n"
                "- Include education level + field when available.\n"
                "- Prefer the current or most recent formal education.\n"
                "- Keep it short and normalized.\n"
                "- Examples:\n"
                "  - \"I completed high school\" -> \"high school\"\n"
                "  - \"I just graduated high school in the Italian islands\" -> \"high school\"\n"
                "  - \"I am completing my master's in clinical psychology\" -> \"master's in clinical psychology\"\n"
                "  - \"I completed my bachelor's and am pursuing a master's in counseling\" -> \"master's in counseling\"\n"
                "  - \"I finished an Istituto Professionale in art and design\" -> \"Istituto Professionale in art and design\"\n"
                "- Do not omit academic_background when formal education is explicitly present.\n"
                "- If no formal education is explicitly stated, omit academic_background.\n"
                "\n"
                + FIELD_OF_INTEREST_RULES +
                "- Update explicit.region ONLY if the student explicitly states one macro-area among North / Center / South / Islands.\n"
                "- Store region ONLY as: north, center, south, islands.\n"
                "- Do NOT infer region from city names, biography details, gender, ethnicity, or socioeconomic cues."
            ),
            schema_example=(
                '<LISTENER_PATCH>{"explicit":{"academic_background":"<short education summary>",'
                '"region":"<north|center|south|islands>"},'
                '"inferred":{"field_of_interest":["<field_of_interest label 1>","<field_of_interest label 2>"]}}</LISTENER_PATCH>'
            ),
            evidence=evidence,
        )

    def _agent_norms_block(self) -> str:
        if not self.agent1_norms:
            return ""
        lines = ["COUNSELOR NORMS FROM SOCIAL PRACTICE:"]
        lines.extend(f"- {norm}" for norm in self.agent1_norms)
        return "\n".join(lines) + "\n"

    # -------------------
    # ------------------------------------------------------
    # Slot sync
    # -------------------------------------------------------------------------
    def _sync_slots_from_listener_memory(self) -> None:
        mem = self._listener_memory if isinstance(self._listener_memory, dict) else {}
        exp = mem.get("explicit", {}) if isinstance(mem.get("explicit"), dict) else {}
        inf = mem.get("inferred", {}) if isinstance(mem.get("inferred"), dict) else {}

        if exp.get("academic_background"):
            self._slots.academic_background = str(exp["academic_background"]).strip()

        reg = exp.get("region")
        reg_norm = _normalize_region_value(reg)
        if reg_norm:
            self._slots.region = reg_norm

        foi = inf.get("field_of_interest")
        if isinstance(foi, list) and foi:
            self._slots.field_of_interest = foi[0]
        elif isinstance(foi, str) and foi.strip():
            self._slots.field_of_interest = foi.strip()

        if exp.get("selected_option"):
            self._slots.selected_option = str(exp["selected_option"]).strip()

    def _displayed_advice_options(self) -> List[str]:
        shown = [str(x).strip() for x in (_LAST_RAG_STATE.advice_options or []) if str(x).strip()]
        return shown or list(self._last_options or [])

    def _update_slots_from_student(self, student_utt: str) -> None:
        while True:
            p = _pop_listener_patch()
            if not isinstance(p, dict):
                break
            self._listener_memory = apply_listener_patch(self._listener_memory, p)

        ab = self._listener_memory.get("explicit", {}).get("academic_background")
        if isinstance(ab, str) and ab.strip():
            lvl = infer_intended_level_from_academic_background(ab)
            if lvl:
                self._slots.intended_level = lvl

        # Selection detection is used only for flow control. The listener memory
        # field explicit.selected_option is still populated only through a
        # <LISTENER_PATCH> generated by the model, not by regex/string rules.
        mo = _OPT_RX.search(student_utt or "")
        if not mo:
            mnum = _NUM_ONLY_RX.search(student_utt or "")
            if mnum:
                mo = mnum

        if mo and self._advice_given:
            n = int(mo.group(1))
            shown_options = self._displayed_advice_options()
            if shown_options and 1 <= n <= len(shown_options):
                self._pending_selection_number = n
                self._last_options = list(shown_options)

        self._sync_slots_from_listener_memory()
        record_debug_snapshot(self._listener_memory, self._slots.summary())

    def _riasec_answer_evidence_block(self) -> str:
        lines = ["RIASEC QUESTION/ANSWER EVIDENCE:"]

        for i, question in enumerate(self.RIASEC_QUESTIONS[:8]):
            answer = self._riasec_answers[i] if i < len(self._riasec_answers) else ""
            lines.append(f"Q{i + 1}: {question}")
            lines.append(f"A{i + 1}: {answer or '[missing]'}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    def _select_blocks_for_options_in_order(
            self,
            raw_blocks: Sequence[str],
            options: Sequence[str],
            max_blocks: Optional[int] = None,
    ) -> List[str]:
        if not raw_blocks or not options:
            return []

        # indicizza i blocchi strutturati per option key
        by_option: Dict[str, List[str]] = defaultdict(list)

        for block in raw_blocks:
            university, course, _ = _parse_program_block(block)
            if not university or not course:
                continue
            key = _option_key(f"{university} | {course}")
            by_option[key].append(block)

        selected: List[str] = []
        seen_blocks = set()

        # mantieni l'ordine delle options mostrate al modello
        for opt in options:
            key = _option_key(opt)
            candidates = by_option.get(key, [])
            for blk in candidates:
                blk_key = _normalize(blk[:400])
                if blk_key in seen_blocks:
                    continue
                seen_blocks.add(blk_key)
                selected.append(blk)
                break

            if max_blocks is not None and len(selected) >= max_blocks:
                break

        return selected

    def _realign_retrieval_to_candidate_pool(
            self,
            retrieval: RetrievalScopeResult,
            candidate_pool: Sequence[str],
    ) -> RetrievalScopeResult:
        aligned_blocks = self._select_blocks_for_options_in_order(
            retrieval.raw_blocks,
            candidate_pool,
            max_blocks=max(4, min(len(candidate_pool), self.candidate_pool_size)),
        )

        # fallback: se per qualche motivo non trova blocchi allineati, non peggiorare
        if not aligned_blocks:
            aligned_blocks = list(
                retrieval.raw_blocks[: max(4, min(len(retrieval.raw_blocks), self.candidate_pool_size))])

        aligned_ctx = _build_short_ctx(
            aligned_blocks,
            self.max_ctx_chars,
            max_blocks=min(len(candidate_pool), 8),
        )
        retrieval.raw_blocks = list(aligned_blocks)
        retrieval.short_ctx = aligned_ctx
        retrieval.options = list(candidate_pool)
        return retrieval


    def _get_ranked_field_of_interest(self) -> List[str]:
        inf = self._listener_memory.get("inferred", {}) if isinstance(self._listener_memory, dict) else {}
        foi = inf.get("field_of_interest")

        if isinstance(foi, list):
            return [x for x in foi if isinstance(x, str) and x.strip()]

        if isinstance(foi, str) and foi.strip():
            return [foi.strip()]

        return []

    def _merge_zone_options_round_robin(
            self,
            zone_results: List[RetrievalScopeResult],
    ) -> List[str]:
        merged: List[str] = []
        seen = set()

        max_len = max((len(r.options) for r in zone_results), default=0)

        for i in range(max_len):
            for res in zone_results:
                if i >= len(res.options):
                    continue

                opt = res.options[i]
                k = _normalize(opt)

                if k in seen:
                    continue

                seen.add(k)
                merged.append(opt)

                if len(merged) >= self.candidate_pool_size:
                    return merged

        return merged

    def _expand_field_label_for_query(self, label: str) -> List[str]:
        return FIELD_QUERY_TERMS.get(label, [label.replace("_", " ")])

    def _build_query_candidates(self, query: str) -> List[str]:
        queries: List[str] = []

        ranked_fields = self._get_ranked_field_of_interest()
        level = (self._slots.intended_level or "").strip()
        academic_background = (self._slots.academic_background or "").strip()

        level_phrase = {
            "bachelor": "bachelor degree program",
            "master": "master degree program",
            "doctorate": "doctoral program",
        }.get(level, "university program")

        # 1) prima query più ricca: background + campo + livello
        if ranked_fields:
            primary = ranked_fields[0]
            queries.append(f"{level_phrase} in {primary}")
            if academic_background:
                queries.append(f"{academic_background} {level_phrase} in {primary}")

        # 2) espansioni disciplinari
        expanded_per_label: List[List[str]] = []
        for label in ranked_fields[:3]:
            terms = self._expand_field_label_for_query(label)
            expanded_per_label.append(terms[:4])

        max_len = max((len(x) for x in expanded_per_label), default=0)

        for i in range(max_len):
            for terms in expanded_per_label:
                if i >= len(terms):
                    continue
                term = terms[i]
                queries.append(term)
                if level:
                    queries.append(f"{level_phrase} in {term}")
                if academic_background:
                    queries.append(f"{academic_background} {term}")

        # 3) fallback finali
        if academic_background:
            queries.append(academic_background)

        if query:
            queries.append(query)

        deduped: List[str] = []
        seen = set()
        for q in queries:
            nq = _normalize(q)
            if nq and nq not in seen:
                seen.add(nq)
                deduped.append(q)

        return deduped

    def _run_retrieval_scope(
            self,
            queries: List[str],
            *,
            allowed_universities: Optional[Sequence[str]] = None,
            section_contains: Optional[str],
            use_preferred_macroarea: bool = True,
            strict_macroarea: bool = False,
    ) -> List[str]:
        allowed_universities = _flatten_to_strings(allowed_universities)

        seen_option_keys = set()
        seen_text_keys = set()
        merged: List[str] = []

        retrieval_pool_limit = max(self.candidate_pool_size * 2, self.top_k * 2, 32)

        for q in queries:
            kwargs: Dict[str, Any] = {
                "top_k": min(retrieval_pool_limit * 2, 40),
            }

            if section_contains:
                kwargs["section_contains"] = section_contains

            if use_preferred_macroarea and self._slots.region:
                kwargs["preferred_macroarea"] = REGION_TO_RAG_MACROAREA.get(
                    self._slots.region,
                    self._slots.region,
                )
                if strict_macroarea:
                    kwargs["strict_macroarea"] = True

            try:
                hits = self.retriever.search(q, **kwargs) or []
            except TypeError:
                # Backward compatibility with retrievers that do not implement strict_macroarea.
                kwargs.pop("strict_macroarea", None)
                hits = self.retriever.search(q, **kwargs) or []

            for h in hits:
                if not isinstance(h, dict):
                    continue

                txt = (h.get("text") or "").strip()
                if not txt:
                    continue

                if _looks_like_catalog_list(txt):
                    continue

                block = _hit_to_program_block(h)
                if not block:
                    continue

                university, course, _ = _parse_program_block(block)

                if not university or not course:
                    _dprint(
                        self.debug,
                        f"[DEBUG][retrieval_scope] reject=no structured university/course "
                        f"meta={(h.get('meta') or {})}"
                    )
                    continue

                if allowed_universities and not _university_matches_allowed(university, allowed_universities):
                    continue

                option_key = _option_key(f"{university} | {course}")
                if option_key:
                    if option_key in seen_option_keys:
                        continue
                    seen_option_keys.add(option_key)
                else:
                    text_key = _normalize(block[:300])
                    if text_key in seen_text_keys:
                        continue
                    seen_text_keys.add(text_key)

                merged.append(block)

                if len(merged) >= retrieval_pool_limit:
                    break

            if len(merged) >= retrieval_pool_limit:
                break

        _dprint(
            self.debug,
            f"[DEBUG][retrieval_scope] queries={len(queries)} "
            f"raw_blocks={len(merged)} "
            f"pool_limit={retrieval_pool_limit} "
            f"capped={len(merged) >= retrieval_pool_limit}"
        )

        return merged

    def _keyword_terms_for_current_profile(self, query: str = "") -> List[str]:
        terms: List[str] = []
        for label in self._get_ranked_field_of_interest()[:3]:
            terms.extend(KEYWORD_FALLBACK_TERMS.get(label, []))
            terms.extend(FIELD_QUERY_TERMS.get(label, [])[:4])

        # Add a few salient words from the query, but avoid broad/noisy words.
        noisy = {"university", "program", "degree", "bachelor", "master", "course", "field", "technology", "technologies"}
        for tok in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", query or ""):
            ntok = _ascii_norm(tok)
            if ntok and ntok not in noisy:
                terms.append(tok)

        deduped: List[str] = []
        seen = set()
        for term in terms:
            t = str(term).strip()
            key = _ascii_norm(t)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(t)
        return deduped[:24]

    def _run_keyword_retrieval_scope(
            self,
            query: str,
            *,
            allowed_universities: Optional[Sequence[str]] = None,
            strict_macroarea: bool = False,
    ) -> List[str]:
        if not hasattr(self.retriever, "keyword_search"):
            return []

        terms = self._keyword_terms_for_current_profile(query)
        if not terms:
            return []

        kwargs: Dict[str, Any] = {
            "keywords": terms,
            "top_k": max(self.candidate_pool_size * 4, self.top_k * 2, 24),
        }

        if self._slots.region:
            kwargs["preferred_macroarea"] = REGION_TO_RAG_MACROAREA.get(self._slots.region, self._slots.region)
            kwargs["strict_macroarea"] = bool(strict_macroarea)

        if allowed_universities:
            kwargs["allowed_universities"] = _flatten_to_strings(allowed_universities)

        try:
            hits = self.retriever.keyword_search(**kwargs) or []
        except TypeError:
            # Backward compatibility with wrappers/retrievers missing optional kwargs.
            kwargs.pop("allowed_universities", None)
            kwargs.pop("strict_macroarea", None)
            hits = self.retriever.keyword_search(**kwargs) or []
        except Exception as exc:
            _dprint(self.debug, f"[DEBUG][keyword_scope] failed: {exc!r}")
            return []

        merged: List[str] = []
        seen = set()
        for h in hits:
            if not isinstance(h, dict):
                continue
            block = _hit_to_program_block(h)
            if not block:
                continue
            university, course, _ = _parse_program_block(block)
            if not university or not course:
                continue
            if allowed_universities and not _university_matches_allowed(university, allowed_universities):
                continue
            key = _option_key(f"{university} | {course}")
            if key in seen:
                continue
            seen.add(key)
            merged.append(block)
            if len(merged) >= max(self.candidate_pool_size * 2, 12):
                break

        _dprint(self.debug, f"[DEBUG][keyword_scope] terms={terms} raw_blocks={len(merged)}")
        return merged

    def _evaluate_scope_result(self, scope_name: str, raw_blocks: List[str]) -> RetrievalScopeResult:
        joined = "\n\n---\n\n".join(raw_blocks).strip()

        ranked_fields = self._get_ranked_field_of_interest()
        raw_options = _extract_ctx_options(
            joined,
            max_items=max(self.candidate_pool_size * 2, 12),
            intended_level=self._slots.intended_level,
            field_of_interest=ranked_fields,
            debug=self.debug,
        )

        options = clean_and_dedup_options(
            raw_options,
            max_options=self.candidate_pool_size * 2,
        )

        options = _filter_options_by_intended_level(
            options,
            self._slots.intended_level,
        )

        options = options[:self.candidate_pool_size]

        _dprint(self.debug, f"[DEBUG][{scope_name}] raw_blocks={len(raw_blocks)}")
        _dprint(self.debug, f"[DEBUG][{scope_name}] raw_extracted_options={len(raw_options)}")
        _dprint(self.debug, f"[DEBUG][{scope_name}] clean_dedup_options={len(options)}")
        for i, opt in enumerate(options[:10], 1):
            _dprint(self.debug, f"[DEBUG][{scope_name}] option_{i}: {opt}")

        aligned_blocks = _select_option_context_blocks(
            raw_blocks,
            options,
            max_blocks=max(4, min(len(options), self.candidate_pool_size))
        )
        short_ctx = _build_short_ctx(aligned_blocks, self.max_ctx_chars)

        return RetrievalScopeResult(
            scope_name=scope_name,
            raw_blocks=list(raw_blocks),
            short_ctx=short_ctx,
            options=options,
        )


    def _get_llm_candidate_pool(self, query: str) -> Tuple[RetrievalScopeResult, List[str]]:
        retrieval = self._retrieve_best_scope(query)

        candidate_pool = clean_and_dedup_options(
            retrieval.options,
            max_options=self.candidate_pool_size,
        )

        candidate_pool = _filter_options_by_intended_level(
            candidate_pool,
            self._slots.intended_level,
        )

        retrieval = self._realign_retrieval_to_candidate_pool(retrieval, candidate_pool)

        _dprint(
            self.debug,
            f"[DEBUG][scope] scope_name={retrieval.scope_name} "
            f"preferred_zone={retrieval.preferred_zone} matched_zone={retrieval.matched_zone}"
        )
        for i, opt in enumerate(candidate_pool, 1):
            _dprint(self.debug, f"[DEBUG][candidate_pool] {i}. {opt}")

        _dprint(self.debug, f"[DEBUG][aligned_ctx_blocks] {len(retrieval.raw_blocks)}")
        for i, blk in enumerate(retrieval.raw_blocks, 1):
            u, c, _ = _parse_program_block(blk)
            _dprint(self.debug, f"[DEBUG][aligned_ctx_block_{i}] {u} | {c}")

        return retrieval, candidate_pool

    def _retrieve_best_scope(self, query: str) -> RetrievalScopeResult:
        queries = self._build_query_candidates(query)
        preferred_zone = _normalize_region_value(self._slots.region)

        def eval_result(scope_name: str, blocks: List[str], matched_zone: Optional[str]) -> RetrievalScopeResult:
            result = self._evaluate_scope_result(scope_name=scope_name, raw_blocks=blocks)
            result.preferred_zone = preferred_zone
            result.matched_zone = matched_zone
            return result

        if preferred_zone:
            preferred_universities = ZONE_TO_UNIVERSITIES.get(preferred_zone, [])

            # 1) Preferred macro-area, vector search, strict metadata.
            raw_blocks = self._run_retrieval_scope(
                queries,
                allowed_universities=None,
                section_contains="Descrizione generale",
                use_preferred_macroarea=True,
                strict_macroarea=True,
            )

            if not raw_blocks:
                raw_blocks = self._run_retrieval_scope(
                    queries,
                    allowed_universities=None,
                    section_contains=None,
                    use_preferred_macroarea=True,
                    strict_macroarea=True,
                )

            # 2) Preferred macro-area, static university aliases fallback.
            if not raw_blocks:
                raw_blocks = self._run_retrieval_scope(
                    queries,
                    allowed_universities=preferred_universities,
                    section_contains="Descrizione generale",
                    use_preferred_macroarea=False,
                )

            if not raw_blocks:
                raw_blocks = self._run_retrieval_scope(
                    queries,
                    allowed_universities=preferred_universities,
                    section_contains=None,
                    use_preferred_macroarea=False,
                )

            if raw_blocks:
                result = eval_result(f"zone_{preferred_zone}", raw_blocks, preferred_zone)
                if result.options:
                    return result

            # 3) Conservative keyword fallback in the preferred macro-area. This
            # recovers cases where vector search misses exact course titles that are
            # clearly present in the RAG, e.g. conservation/environmental programs.
            keyword_blocks = self._run_keyword_retrieval_scope(
                query,
                allowed_universities=preferred_universities,
                strict_macroarea=True,
            )
            if keyword_blocks:
                result = eval_result(f"zone_{preferred_zone}_keyword", keyword_blocks, preferred_zone)
                if result.options:
                    return result

        # 4) National fallback. Keep the preferred macro-area as a soft scoring
        # bonus but do not claim that outside-area options are local.
        raw_blocks = self._run_retrieval_scope(
            queries,
            allowed_universities=None,
            section_contains="Descrizione generale",
            use_preferred_macroarea=True,
        )

        if not raw_blocks:
            raw_blocks = self._run_retrieval_scope(
                queries,
                allowed_universities=None,
                section_contains=None,
                use_preferred_macroarea=True,
            )

        if raw_blocks:
            result = eval_result("italy_fallback", raw_blocks, None)
            if result.options:
                return result

        # 5) National keyword fallback as a last resort, still grounded in RAG rows.
        keyword_blocks = self._run_keyword_retrieval_scope(
            query,
            allowed_universities=None,
            strict_macroarea=False,
        )
        if keyword_blocks:
            result = eval_result("italy_keyword_fallback", keyword_blocks, None)
            if result.options:
                return result

        empty = RetrievalScopeResult()
        empty.preferred_zone = preferred_zone
        empty.matched_zone = None
        return empty

    # -------------------------------------------------------------------------
    # Prompt/query helpers
    # -------------------------------------------------------------------------
    def _build_query(self) -> str:
        level = (self._slots.intended_level or "").strip()
        ranked_fields = self._get_ranked_field_of_interest()

        level_phrase = {
            "bachelor": "bachelor degree program",
            "master": "master degree program",
            "doctorate": "doctoral program",
        }.get(level, "university program")

        if ranked_fields:
            field_phrase = " / ".join(ranked_fields[:3])
        else:
            field_phrase = (self._slots.field_of_interest or "").strip()

        if level and field_phrase:
            return f"{level_phrase} in {field_phrase}"
        if field_phrase:
            return field_phrase
        if level:
            return level_phrase
        return "university degree program in Italy"


    def _option_metadata_block(self, options: List[str]) -> str:
        lines = []
        for i, opt in enumerate(options, start=1):
            university = opt.split("|", 1)[0].strip() if "|" in opt else opt.strip()
            macro_area = _infer_zone_from_university(university) or "unknown"
            lines.append(f"- Option {i}: macro_area={macro_area}; university={university}")
        return "\n".join(lines)

    def _classify_question(self, t: str) -> str:
        s = (t or "").strip()
        low = s.lower()

        if "?" not in s:
            return "other"
        if "why" in low:
            return "why"

        if re.search(r"\b(like|such as|e\.g\.|for example)\b.*\bor\b", low):
            if self._AUX_YN_RX.search(low):
                return "yesno"
            return "other"

        if re.search(
                r"\b("
                r"would you recommend|would you suggest|do you recommend|do you suggest|"
                r"should i|which is better|prefer|"
                r"mi consiglieresti|mi suggeriresti|consigli|suggerisci|"
                r"meglio|conviene"
                r")\b.*\b(or|oppure|o)\b",
                low,
        ):
            return "choice"

        if ("tra " in low and " e " in low) or ("between" in low and " and " in low):
            return "choice"

        if ("between" in low and "and" in low):
            return "choice"

        if self._WH_RX.search(low) or self._REQ_RX.search(low):
            return "other"

        if self._AUX_YN_RX.search(low):
            return "yesno"

        return "other"

    def _extract_choice_options(self, q: str) -> List[str]:
        qq = (q or "").strip()
        low = qq.lower()

        # Prefer the configured fixed follow-up questions from social_practices.json.
        # This keeps the counselor-side choice handling aligned with the student script.
        q_norm = _normalize(qq)
        for idx, pair in CHOICE_REPLY_OPTIONS.get(self.lang, {}).items():
            if idx < len(self.FIXED_FOLLOWUP_QUESTIONS):
                configured_q = self.FIXED_FOLLOWUP_QUESTIONS[idx]
                if q_norm == _normalize(configured_q):
                    return list(pair)

        if ("percorso tecnico" in low) and ("discipline umanistiche" in low):
            return [
                "concentrarti su un percorso tecnico",
                "un percorso orientato alle discipline umanistiche",
            ]

        if ("ruolo di leadership" in low) and ("compiti di supporto" in low):
            return [
                "assumere già ora un ruolo di leadership",
                "concentrarti prima su compiti di supporto",
            ]

        if ("corsi avanzati" in low) and ("moduli introduttivi" in low):
            return [
                "iniziare subito con corsi avanzati",
                "partire da moduli introduttivi",
            ]

        qq = re.sub(r"[\s\?\.]+$", "", qq)

        if ("technical program" in low) and ("program oriented toward humanities" in low):
            return [
                "focusing on a technical program",
                "a program oriented toward humanities",
            ]

        if ("leadership role now" in low) and ("supporting tasks first" in low):
            return [
                "take a leadership role now",
                "focus on supporting tasks first",
            ]

        if ("advanced classes immediately" in low) and ("introductory modules" in low):
            return [
                "take advanced classes immediately",
                "start with introductory modules",
            ]

        parts = self._CHOICE_CONNECTOR.split(qq)
        if len(parts) >= 2:
            return [parts[-2].strip(), parts[-1].strip()][:2]

        m = re.search(r"\bbetween\b\s+(.+?)\s+\band\b\s+(.+)$", qq, re.I)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]

        return []

    def _extract_last_question_line(self, text: str) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        for ln in reversed(lines):
            if "?" in ln:
                return ln
        return (text or "").strip()

    def _qa_context_block(self) -> str:
        exp = self._listener_memory.get("explicit", {}) if isinstance(self._listener_memory, dict) else {}
        inf = self._listener_memory.get("inferred", {}) if isinstance(self._listener_memory, dict) else {}

        def clean(x: Any) -> Optional[str]:
            if isinstance(x, str):
                s = x.strip()
                return s if s else None
            return None

        foi_value = inf.get("field_of_interest")
        if isinstance(foi_value, list) and foi_value:
            foi_rendered = " > ".join(str(x).strip() for x in foi_value if str(x).strip())
        else:
            foi_rendered = clean(foi_value)

        riasec_value = inf.get("riasec_attitudes")
        if isinstance(riasec_value, list) and riasec_value:
            riasec_rendered = ", ".join(str(x).strip() for x in riasec_value if str(x).strip())
        else:
            riasec_rendered = None

        selected_value = clean(exp.get("selected_option"))
        shown_options = self._displayed_advice_options()
        if selected_value:
            selected_value = _canonicalize_selected_option_value(selected_value, shown_options)
        if not selected_value and self._pending_selection_number and shown_options:
            n = self._pending_selection_number
            if 1 <= n <= len(shown_options):
                selected_value = f"{n} - {shown_options[n - 1]}"

        fields = [
            ("academic_background", clean(exp.get("academic_background"))),
            ("field_of_interest", foi_rendered),
            ("region", clean(exp.get("region"))),
            ("selected_option", selected_value),
            ("riasec_attitudes", riasec_rendered),
        ]

        if os.getenv("SDIALOG_EXPOSE_SENSITIVE_CONTEXT", "0").lower() in {"1", "true", "yes"}:
            fields.append(("gender", clean(inf.get("gender"))))

        lines = [f"- {k}: {v}" for k, v in fields if v is not None]
        if not lines:
            return "CONTEXT (use internally to decide; do NOT restate unless asked):\n"
        return "CONTEXT (use internally to decide; do NOT restate unless asked):\n" + "\n".join(lines) + "\n"

    def _build_advice_fallback(self) -> Tuple[str, str, bool]:
        message = (
            "I could not find grounded options that fit your current profile closely enough in the current university set. "
            "Please clarify one point only: should I broaden toward adjacent fields, or keep the field exact and prioritize your current constraints?"
        )
        return message, "", False

    def _search_note_for_scope(self, scope_name: str) -> str:
        if not self._slots.region:
            return ""

        if scope_name.startswith("zone_"):
            return ""

        if scope_name in {"italy_fallback", "italy_keyword_fallback"}:
            return (
                f"NOTE: no grounded option was found in the preferred area ({self._slots.region}). "
                "The OPTIONS below are grounded alternatives from the current university set across Italy.\n\n"
            )

        return ""

    # -------------------------------------------------------------------------
    # Main orchestration
    # -------------------------------------------------------------------------
    def instruct(self, dialog, utterance):
        student_utt = (utterance or "").strip()

        if self._finished:
            update_last_rag_state("", query="", phase="stop", expected_verbatim="STOP")
            return "Output ONLY: STOP"

        if not student_utt:
            turns = getattr(dialog, "turns", None) or []
            for t in reversed(turns):
                txt = _safe_turn_text(t)
                if txt:
                    student_utt = txt
                    break

        if student_utt:
            self._update_slots_from_student(student_utt)

        if self._flow_stage == "riasec" and len(self._riasec_answers) < 8:
            self._riasec_answers.append(student_utt)

        if self._advice_given and _CLOSE_RX.search(student_utt or ""):
            update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="wrapup", advice_options=self._displayed_advice_options())

            riasec_labels = (self._listener_memory.get("inferred", {}) or {}).get("riasec_attitudes")
            code = _riasec_labels_to_code(riasec_labels) or "UNK"

            self._finished = True
            return (
                f"PHASE: WRAPUP\n"
                f"RIASEC: {code}\n"
                "Write the final message using EXACTLY this multiline structure:\n"
                "Recap: <1-2 short sentences on background + selected-option fit>\n"
                f"RIASEC: {code}\n"
                "Next steps:\n"
                "1. <review official course plan/admission requirements>\n"
                "2. <verify costs/location/commuting or housing constraints>\n"
                "3. <prepare documents, portfolio, or prerequisite review>\n"
                "Goodbye: <short friendly goodbye>\n"
                "Rules for next steps:\n"
                "- Do NOT introduce new universities/programs.\n"
                "- Do NOT invent exact deadlines, orientation sessions, coordinators, networks, or services unless they were in the retrieved context.\n"
                "- Do NOT omit item 1, 2, or 3.\n"
            )
        if self._advice_given and (self._slots.selected_option or self._pending_selection_number) and self._flow_stage != "qa":
            self._flow_stage = "qa"
            self._pending_selection_ack = ("?" not in (student_utt or ""))

        # ---------------------------------------------------------------------
        # Q&A
        # ---------------------------------------------------------------------
        if self._flow_stage == "qa":
            selection_patch = ""
            if (self._pending_selection_number is not None) and (self._selection_patch_requested_for != self._pending_selection_number):
                self._selection_patch_requested_for = self._pending_selection_number
                shown_options = self._displayed_advice_options()
                shown_block = "\n".join(f"{i}. {o}" for i, o in enumerate(shown_options, start=1))
                selection_patch = self._listener_patch_request(
                    targets="- explicit.selected_option",
                    rules=(
                        "- selected_option MUST be exactly in the format 'N - <EXACT option string from ADVICE_OPTIONS_SHOWN_TO_STUDENT>'.\n"
                        "- Do NOT paraphrase.\n"
                        "- Use the option number chosen by the student and the exact option text from ADVICE_OPTIONS_SHOWN_TO_STUDENT.\n"
                        "- If the student did not clearly choose an option number, omit selected_option."
                    ),
                    schema_example=(
                        '<LISTENER_PATCH>{"explicit":{"selected_option":'
                        '"1 - University ... | [CODE] COURSE"}}'
                        "</LISTENER_PATCH>"
                    ),
                    evidence=(
                        f"STUDENT_SELECTION_MESSAGE:\n{student_utt}\n\n"
                        f"ADVICE_OPTIONS_SHOWN_TO_STUDENT:\n{shown_block}"
                    ),
                )

            def _ret(msg: str) -> str:
                return (msg + selection_patch) if selection_patch else msg

            if self._pending_selection_ack:
                self._pending_selection_ack = False
                shown_options = self._displayed_advice_options()
                update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_selected", advice_options=shown_options)
                selected_for_ack = self._slots.selected_option
                if selected_for_ack:
                    selected_for_ack = _canonicalize_selected_option_value(selected_for_ack, shown_options)
                if not selected_for_ack and self._pending_selection_number and shown_options:
                    n = self._pending_selection_number
                    if 1 <= n <= len(shown_options):
                        selected_for_ack = f"{n} - {shown_options[n - 1]}"
                return _ret(
                    "PHASE: Q&A\n"
                    f"The student selected: {selected_for_ack or 'the chosen option'}\n"
                    "TASK:\n"
                    "- Acknowledge the selection in ONE short sentence.\n"
                    "- Do NOT introduce new universities/programs.\n"
                    "- Do NOT ask any questions.\n"
                )

            q_for_qa = self._extract_last_question_line(student_utt)
            qtype = self._classify_question(q_for_qa)

            if qtype == "choice":
                ch = self._extract_choice_options(q_for_qa)
                update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_choice", choice_options=ch, advice_options=self._displayed_advice_options())
                if len(ch) == 2:
                    return _ret(
                        "PHASE: Q&A\n"
                        "QUESTION TYPE: CHOICE, not yes/no.\n"
                        + self._qa_context_block() +
                        "TASK:\n"
                        f"- Reply with ONLY ONE of these exact strings: '{ch[0]}' OR '{ch[1]}'.\n"
                        "- Do NOT answer Yes or No.\n"
                        "- Output EXACTLY that selected string and nothing else.\n"
                    )
                return _ret(
                    "PHASE: Q&A\n"
                    "QUESTION TYPE: CHOICE, not yes/no.\n"
                    + self._qa_context_block() +
                    "TASK:\n"
                    "- Reply with ONLY one of the two options mentioned in the student's question.\n"
                    "- Do NOT answer Yes or No.\n"
                    "- Do NOT add any extra words.\n"
                )

            if qtype == "why":
                update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_why", advice_options=self._displayed_advice_options())
                return _ret(
                    "PHASE: Q&A\n"
                    + self._agent_norms_block()
                    + self._qa_context_block() +
                    "TASK:\n"
                    "- Answer the student's 'why' briefly.\n"
                    "- Ground your justification in the retrieved context; do not invent program details.\n"
                    "- Use the student's background + constraints + RIASEC to explain trade-offs.\n"
                    "- Do NOT introduce new universities/programs.\n"
                )

            if qtype == "yesno":
                update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_yesno", advice_options=self._displayed_advice_options())
                return _ret(
                    "PHASE: Q&A\n"
                    "QUESTION TYPE: YES/NO suitability/feasibility/readiness/risk.\n"
                    + self._qa_context_block() +
                    "TASK:\n"
                    "- Reply with ONLY 'Yes' or 'No'.\n"
                    "- Decide using the CONTEXT above (background, constraints, RIASEC, and the selected option).\n"
                )

            update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_other", advice_options=self._displayed_advice_options())
            return _ret(
                "PHASE: Q&A\n"
                + self._agent_norms_block()
                + self._qa_context_block() +
                "TASK:\n"
                "- Answer briefly and pragmatically.\n"
                "- Do NOT introduce new universities/programs.\n"
            )

        # ---------------------------------------------------------------------
        # GREET
        # ---------------------------------------------------------------------
        if self._flow_stage == "greet":
            q = self.BG_QUESTIONS[0]
            self._flow_stage = "background"
            self._bg_i = 0
            update_last_rag_state("", phase="greet", expected_verbatim=q)

            patch = self._listener_patch_request(
                targets="- explicit.region\n- inferred.field_of_interest\n- inferred.gender",
                rules=(
                    "Rules for explicit.region:\n"
                    "- Store region only if the student explicitly mentions a macro-area of Italy.\n"
                    "- Allowed values: north, center, south, islands.\n"
                    "- Map 'Northern Italy' -> north, 'Central Italy' -> center, 'Southern Italy' -> south.\n"
                    "- Map 'Italian islands', 'the Italian islands', 'Sicily', 'Sardinia' -> islands.\n"
                    "- Do not infer region from cities, towns, provinces, or vague descriptions.\n"
                    "\n"
                    + FIELD_OF_INTEREST_RULES +
                    "Rules for inferred.gender:\n"
                    "- Infer gender only if there is a clear cue in the student's message.\n"
                    "- If there is no clear cue, omit the field."
                ),
                schema_example=(
                    '<LISTENER_PATCH>{"explicit":{"region":"<inferred region>"},'
                    '"inferred":{"field_of_interest":["<field_of_interest label 1>","<field_of_interest label 2>","<field_of_interest label 3>"],'
                    '"gender":"<gender>"}}</LISTENER_PATCH>'
                ),
                evidence="Use the student's FIRST message in this conversation.",
            )

            return (
                "PHASE: GREET & ROLE\n"
                "TASK:\n"
                "- Greet the student.\n"
                "- State your role: you help students choose university programs based on background, interests, aptitudes, and constraints.\n"
                "- Then ask ONE question (verbatim):\n"
                f"{q}\n"
                + patch
            )

        # ---------------------------------------------------------------------
        # BACKGROUND
        # ---------------------------------------------------------------------
        if self._flow_stage == "background":
            self._bg_i += 1
            if self._bg_i < len(self.BG_QUESTIONS):
                q = self.BG_QUESTIONS[self._bg_i]
                update_last_rag_state("", phase=f"background_q{self._bg_i + 1}", expected_verbatim=q)
                patch = self._background_slot_patch_request(
                    "Use the student's latest answer and any earlier background answers already given in this conversation."
                )
                return (
                    "PHASE: ACQUIRE ACADEMIC BACKGROUND\n"
                    "Visible reply: respond naturally in 1–2 short sentences, then ask ONE question verbatim.\n"
                    "After the visible reply, append the hidden listener patch.\n"
                    f"Question to ask verbatim:\n{q}\n"
                    + patch
                )
            self._flow_stage = "riasec"
            self._riasec_i = 0

        # ---------------------------------------------------------------------
        # RIASEC
        # ---------------------------------------------------------------------
        if self._flow_stage == "riasec":
            if self._riasec_i < len(self.RIASEC_QUESTIONS):
                q = self.RIASEC_QUESTIONS[self._riasec_i]
                self._riasec_i += 1
                update_last_rag_state("", phase=f"riasec_q{self._riasec_i}", expected_verbatim=q)

                bg_patch = ""

                if self._riasec_i == 1:
                    bg_patch = self._listener_patch_request(
                        targets="- explicit.academic_background\n- inferred.field_of_interest\n- explicit.region",
                        rules=(
                            "Rules for explicit.academic_background:\n"
                            "- If the student states any current or completed formal education, you MUST include academic_background.\n"
                            "- academic_background must summarize only formal education, not jobs, hobbies, internships, goals, or personal interests.\n"
                            "- Include education level + field when available.\n"
                            "- Prefer the current or most recent formal education.\n"
                            "- Keep it short and normalized.\n"
                            "- Examples:\n"
                            "  - \"I am completing my master's in clinical psychology\" -> \"master's in clinical psychology\"\n"
                            "  - \"I just finished Liceo Classico\" -> \"Liceo Classico\"\n"
                            "  - \"I graduated from a Liceo Scientifico\" -> \"Liceo Scientifico\"\n"
                            "  - \"I'm pursuing a bachelor's in environmental science\" -> \"bachelor's in environmental science\"\n"
                            "  - \"I finished an Istituto Professionale in art and design\" -> \"Istituto Professionale in art and design\"\n"
                            "- Do not omit academic_background when formal education is explicitly present.\n"
                            "- If no formal education is explicitly stated, omit academic_background.\n"
                            "\n"
                            + FIELD_OF_INTEREST_RULES +
                            "- Update explicit.region ONLY if the student explicitly states one macro-area among North / Center / South / Islands.\n"
                            "- Store region ONLY as: north, center, south, islands.\n"
                            "- Do NOT infer region from city names or biography details."
                        ),
                        schema_example=(
                            '<LISTENER_PATCH>{"explicit":{"academic_background":"<short education summary>",'
                            '"region":"<north|center|south|islands>"},'
                            '"inferred":{"field_of_interest":["<field_of_interest label 1>","<field_of_interest label 2>","<field_of_interest label 3>"]}}</LISTENER_PATCH>'
                        ),
                        evidence="Use the student's answers to the background questions already given in this conversation.",
                    )

                return (
                    "PHASE: GATHER APTITUDES (RIASEC)\n"
                    "Visible reply: ask the following question VERBATIM. Ask ONE visible question only.\n"
                    "After the visible question, append the hidden listener patch if requested.\n"
                    f"{q}\n"
                    + bg_patch
                )

            self._flow_stage = "advice"

        # ---------------------------------------------------------------------
        # ADVICE
        # ---------------------------------------------------------------------
        if not self._slots.academic_background:
            message = _t(
                "Before I recommend programs, could you clarify your current or most recent formal education?",
                "Prima di consigliarti dei corsi, puoi chiarire qual è il tuo percorso di istruzione formale attuale o più recente?"
            )

            update_last_rag_state(
                "",
                query="",
                phase="clarify_scope",
                expected_verbatim=message,
                advice_fallback=True,
                advice_options=[],
            )

            patch = self._background_slot_patch_request(
                "Use the student's latest clarification answer if it contains formal education; otherwise omit unknown fields."
            )
            return (
                "PHASE: CLARIFY_SCOPE\n"
                "Visible reply: ask ONE short question only.\n"
                "After the visible question, append the hidden listener patch.\n"
                f"{message}\n"
                + patch
            )
        query = self._build_query()
        retrieval, candidate_pool = self._get_llm_candidate_pool(query)

        self._last_ctx = retrieval.short_ctx
        self._last_options = list(candidate_pool)

        riasec_patch = ""
        if not self._riasec_patch_requested:
            self._riasec_patch_requested = True
            riasec_patch = self._listener_patch_request(
                targets="- inferred.riasec_question_valences",
                rules=(
                    "Use ONLY the student's answers to the 8 RIASEC questions.\n"
                    "Do NOT use biography, gender, academic background, grades, job, goals, "
                    "field_of_interest, selected_option, or university options.\n"
                    "\n"
                    "Your task is to estimate the student's expressed INTEREST/PREFERENCE "
                    "for each RIASEC question.\n"
                    "Do NOT estimate ability, confidence, preparation, social desirability, "
                    "or academic readiness.\n"
                    "\n"
                    "Return exactly one JSON object inside <LISTENER_PATCH> ... </LISTENER_PATCH>.\n"
                    "Return only the requested field.\n"
                    "\n"
                    "Return an array of exactly 8 integers named riasec_question_valences.\n"
                    "Each integer corresponds to Q1..Q8 in order.\n"
                    "\n"
                    "Scale:\n"
                    "- 2 = clear and strong liking, with explicit enthusiasm or concrete examples\n"
                    "- 1 = moderate liking, curiosity, or generally positive attitude\n"
                    "- 0 = unclear, mixed, conditional, only ability mentioned, or insufficient evidence\n"
                    "- -1 = moderate dislike, avoidance, or low preference\n"
                    "- -2 = clear and strong dislike or rejection\n"
                    "\n"
                    "Calibration rules:\n"
                    "- If the student likes an activity but feels insecure about being good at it, "
                    "score the interest as positive.\n"
                    "- If the student says they are good at an activity but do not enjoy it, "
                    "score it neutral or negative.\n"
                    "- Do not treat long or enthusiastic wording as strong evidence unless it is "
                    "about the specific activity in the question.\n"
                    "- Do not infer a RIASEC type from career goals, school background, grades, "
                    "gender, or biography.\n"
                    "- Do not force a positive score: neutral or unclear answers should be 0.\n"
                    "- Ambivalent answers such as 'sometimes', 'it depends', 'maybe', or 'a little' "
                    "are usually 0 or 1, not 2.\n"
                    "- Strong rejection such as 'I really do not like it', 'I avoid it', or "
                    "'I would not want to do that' is -2.\n"
                    "\n"
                    "Question mapping used later by Python:\n"
                    "- Q1 practical, hands-on work, building or repairing things -> Realistic\n"
                    "- Q2 understanding how things work -> Investigative; possible Realistic nuance "
                    "only if the answer mentions concrete mechanisms, tools, objects, or systems\n"
                    "- Q3 researching, analyzing problems, logical reasoning -> Investigative\n"
                    "- Q4 creative expression through writing, art, music, or design -> Artistic\n"
                    "- Q5 experimenting and creating new things -> Artistic / Investigative / Enterprising nuance; "
                    "score the general preference for experimenting and creating\n"
                    "- Q6 working closely with people in a supportive or helping role -> Social\n"
                    "- Q7 proposing ideas and organizing projects -> Enterprising / Conventional nuance\n"
                    "- Q8 keeping everything organized and under control -> Conventional\n"
                    "\n"
                    "Do NOT output final RIASEC labels. Python will calculate the top 3 labels "
                    "and confidence from riasec_question_valences.\n"
                ),
                schema_example=(
                    '<LISTENER_PATCH>{"inferred":{"riasec_question_valences":[0,0,0,0,0,0,0,0]}}</LISTENER_PATCH>'
                ),
                evidence=self._riasec_answer_evidence_block(),
            )
        if retrieval.is_empty or not candidate_pool:
            self._advice_retry_count += 1

            if self._advice_retry_count <= 1:
                message, fallback_patch, _ = self._build_advice_fallback()

                update_last_rag_state(
                    retrieval.short_ctx,
                    query=query,
                    phase="clarify_scope",
                    expected_verbatim=message,
                    advice_fallback=True,
                    advice_options=[],
                )

                return (
                        "PHASE: CLARIFY_SCOPE\n"
                        "No grounded options are available yet.\n"
                        "Ask ONE short clarification question only.\n"
                        + fallback_patch
                        + riasec_patch
                )

            self._finished = True
            self._flow_stage = "done"
            update_last_rag_state(
                "",
                query=query,
                phase="done",
                expected_verbatim="",
                advice_fallback=False,
                advice_options=[],
            )
            return (
                "PHASE: DONE\n"
                "This is the final counselor message.\n"
                "- Briefly state that no grounded options were found in the current university set.\n"
                "- Give the next steps in one prose sentence, not as a numbered or bulleted list.\n"
                "- Use only these ideas: broaden the target field slightly; keep the field exact and search again later when more universities are available; review constraints/priorities.\n"
                "- Then say goodbye.\n"
                "- Do NOT ask any further question.\n"
                "- Do NOT continue the conversation beyond this message.\n"
            )

        self._advice_retry_count = 0
        self._advice_given = True

        update_last_rag_state(
            retrieval.short_ctx,
            query=query,
            phase="advice",
            advice_fallback=False,
            advice_options=candidate_pool,
        )

        ranked_fields = self._get_ranked_field_of_interest()
        field_profile = " > ".join(ranked_fields) if ranked_fields else (self._slots.field_of_interest or "unknown")

        riasec_value = (self._listener_memory.get("inferred", {}) or {}).get("riasec_attitudes")
        if isinstance(riasec_value, list) and riasec_value:
            riasec_profile = ", ".join(str(x).strip() for x in riasec_value if str(x).strip())
        else:
            riasec_profile = "unknown"

        student_profile_block = (
            "STUDENT PROFILE:\n"
            f"- academic_background: {self._slots.academic_background or 'unknown'}\n"
            f"- field_of_interest: {field_profile}\n"
            f"- intended_level: {self._slots.intended_level or 'unknown'}\n"
            f"- region: {self._slots.region or 'unknown'}\n"
            f"- riasec_attitudes: {riasec_profile}\n"
        )

        opt_lines = "\n".join([f"- Option {i + 1}: {o}" for i, o in enumerate(candidate_pool)])
        option_meta = self._option_metadata_block(candidate_pool)
        search_note = self._search_note_for_scope(retrieval.scope_name)

        final_line = _advice_final_line()
        internal_riasec_step = ""
        riasec_evidence_for_ranking = ""
        if riasec_profile == "unknown" and len(self._riasec_answers) >= 8:
            riasec_evidence_for_ranking = (
                "RIASEC ANSWERS FOR RANKING:\n"
                f"{self._riasec_answer_evidence_block()}\n\n"
            )
        if riasec_profile == "unknown":
            internal_riasec_step = (
                "INTERNAL RIASEC STEP (do not show):\n"
                "- Infer the student's top 3 RIASEC labels from the 8 RIASEC answers in the conversation.\n"
                "- Use those inferred labels when ranking the candidate options.\n"
                "- The RIASEC labels used for ranking MUST be the same labels emitted in <LISTENER_PATCH>.\n"
                "- Do not use any heuristic or fixed scoring; infer from the student's natural-language answers.\n\n"
            )
        return (
            "PHASE: ADVICE\n"
            + self._agent_norms_block()
            + student_profile_block + "\n"
            + search_note
            + "CANDIDATE OPTIONS (already filtered for basic compatibility):\n"
            f"{opt_lines}\n\n"
            + "OPTION METADATA:\n"
            f"{option_meta}\n\n"
              "RAG_CONTEXT FOR THE CANDIDATE OPTIONS (paraphrase only; do not copy markers/IDs):\n"
              f"{retrieval.short_ctx}\n\n"
            + internal_riasec_step
            + riasec_evidence_for_ranking
            +"TASK:\n"
            "- Use ONLY the options listed in CANDIDATE OPTIONS.\n"
            "- These options are already filtered for basic compatibility, deduplicated, and cleaned.\n"
            "- The candidate options are not necessarily ordered. "
            "- Rank them using the student profile, including the inferred RIASEC labels.\n"
            "- Use the student profile to decide the ranking.\n"
            "- Prefer matches with the FIRST field_of_interest label over matches with later labels.\n"
            "- Treat the student's stated macro-area as a real constraint for ranking.\n"
            "- If same-area options are present, rank same-area options before outside-area options.\n"
            "- If the search note says no grounded option was found in the preferred area, do not describe outside-area options as near home or region-compatible.\n"
            "- Do NOT invent new programs, universities, course details, rankings, or explanations.\n"
            "- Return up to 3 options, only if they are genuinely plausible.\n"
            "- If no candidate is a plausible fit, state that no grounded option is available.\n"
            "\n"
            "OUTPUT RULES:\n"
            "- Output ONLY the final options, up to 3.\n"
            "- Use the exact option text from CANDIDATE OPTIONS.\n"
            "- Do not add explanations, comments, headers, bullets, or extra prose.\n"
            "- Output format must be numbered, one option per line:\n"
            "  1. <exact option text>\n"
            "  2. <exact option text>\n"
            "  3. <exact option text if available>\n"
            f'  Final line: "{final_line}"\n'
            + riasec_patch
        )

    def can_finish(self, utterance: str) -> bool:
        return self._finished


