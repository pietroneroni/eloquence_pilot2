from __future__ import annotations

import copy
import json
import re
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

_MAX_ADVICE_RETRIES = 2

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

# =============================================================================
# Generic helpers
# =============================================================================
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().casefold()

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

def _extract_university_from_block(text: str) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("UNIVERSITÀ:"):
            value = line.split(":", 1)[1].strip()
            return value or None
    return None

def _normalize_university_name(value: str) -> str:
    return _normalize(value).replace("universita", "università")


def _hit_university_name(hit: Dict[str, Any]) -> str:
    meta = hit.get("meta") or {}
    for key in ("UNIVERSITY", "UNIVERSITÀ", "UNIVERSITA", "university", "ateneo"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    txt = (hit.get("text") or "").strip()
    uni = _extract_university_from_block(txt)
    return uni or ""


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


def _riasec_code_to_labels(code: str) -> List[str]:
    mapping = {
        "R": "Realistic",
        "I": "Investigative",
        "A": "Artistic",
        "S": "Social",
        "E": "Enterprising",
        "C": "Conventional",
    }
    out: List[str] = []
    for ch in (code or "").upper():
        if ch in mapping and mapping[ch] not in out:
            out.append(mapping[ch])
    return out[:3]



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
    out: List[Tuple[int, str]] = []
    for line in (answer or "").splitlines():
        s = line.strip()
        m = re.match(r"^([1-3])\.\s+(.+?)\s*$", s)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def _validate_advice_answer(answer: str, allowed_options: Sequence[str]) -> Optional[str]:
    lines = _extract_numbered_advice_lines(answer)
    if not lines:
        return None

    max_n = min(3, len(allowed_options or []))
    lines = lines[:max_n]

    expected_nums = list(range(1, len(lines) + 1))
    if [n for n, _ in lines] != expected_nums:
        return None

    allowed_map = {_normalize(opt): opt for opt in (allowed_options or []) if isinstance(opt, str) and opt.strip()}
    chosen: List[str] = []
    seen = set()

    for _, text in lines:
        key = _normalize(text)
        if key not in allowed_map:
            return None
        canon = allowed_map[key]
        ck = _normalize(canon)
        if ck in seen:
            return None
        seen.add(ck)
        chosen.append(canon)

    if not chosen:
        return None

    final_line = _advice_final_line()
    return "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(chosen)] + [final_line])

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
            "comunicazione", "communication", "media",
            "architettura", "architecture",
            "progetto", "progettazione",
        ],
        "design": [
            "design", "visual", "grafica", "communication", "comunicazione",
            "media", "architecture", "architettura", "progetto", "progettazione",
        ],
        "communication_digital_media": [
            "communication", "comunicazione", "media", "digital", "digitale",
            "visual", "grafica", "design",
        ],
        "humanities": [
            "lettere", "literature", "literary", "philology", "filologia",
            "philosophy", "filosofia",
            "history", "historical", "storia", "studies",
            "languages", "lingue", "language",
            "cultural", "culture", "cultura",
            "beni culturali", "humanities", "umanistiche",
        ],
        "cultural_heritage": [
            "beni culturali", "cultural heritage", "heritage", "patrimonio culturale",
            "art history", "storia dell'arte", "archeology", "archaeology",
            "archeologia", "museum", "musei", "cultural", "culture",
        ],
        "social_sciences": [
            "sociology", "sociologia", "social", "political", "politics",
            "international studies", "global", "local studies",
        ],
        "environmental_science": [
            "environment", "environmental", "ambiente", "ambientale",
            "ecology", "ecologia", "ecosystem", "ecosystems",
            "territorio", "paesaggio", "sustainability", "sostenibil",
        ],
        "biology": [
            "biology", "biologia", "biological", "biologico", "biologica",
            "biotechnology", "biotecnologie", "scienze biologiche",
        ],
        "computer_science": [
            "computer science", "informatica", "software", "data",
            "artificial intelligence", "intelligenza artificiale",
        ],
    }

    hard_negative = {
        "arts_design": [
            "medicine", "medicina", "nursing", "infermieristica",
            "pharmacy", "farmacia", "odontoiatria",
            "chemistry", "chimica", "physics", "fisica",
            "economics", "finance", "statistica", "law", "giurisprudenza",
        ],
        "design": [
            "medicine", "medicina", "pharmacy", "farmacia",
            "chemistry", "chimica", "physics", "fisica",
            "economics", "law", "giurisprudenza",
        ],
        "humanities": [
            "engineering", "ingegneria", "computer science", "informatica",
            "medicine", "medicina", "pharmacy", "farmacia",
            "biotechnology", "biotecnologie",
        ],
        "cultural_heritage": [
            "medicine", "medicina", "pharmacy", "farmacia",
            "computer science", "informatica pura",
            "biotechnology", "biotecnologie",
        ],
        "environmental_science": [
            "odontoiatria", "pharmacy", "farmacia", "law", "giurisprudenza",
            "economics", "economia pura",
        ],
    }

    # Hard negative: se il corso è chiaramente fuori dominio per tutti i campi, scarta.
    for field in fields:
        for kw in hard_negative.get(field, []):
            if kw in c:
                return True

    # Positive match: basta che il corso matchi almeno uno dei campi inferiti.
    for field in fields:
        kws = positive.get(field, [])
        if any(kw in c for kw in kws):
            return False

    # Per campi creativi/umanistici/sociali, se non c'è nessun segnale positivo, scarta.
    strict_fields = {
        "arts_design",
        "design",
        "communication_digital_media",
        "humanities",
        "cultural_heritage",
        "social_sciences",
        "environmental_science",
        "biology",
        "computer_science",
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

        if _course_domain_mismatch(course, field_of_interest):
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
        },
    }


def _empty_listener_memory() -> dict:
    return canonical_listener_memory()

def _normalize_field_of_interest(value: Any) -> Optional[List[str]]:
    allowed = {
        "arts_design",
        "design",
        "communication_digital_media",
        "cultural_heritage",
        "humanities",
        "psychology",
        "cognitive_science_linguistics",
        "social_sciences",
        "biology",
        "environmental_science",
        "environmental_engineering",
        "computer_science",
    }

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
        exp["selected_option"] = pexp["selected_option"].strip()

    foi = _normalize_field_of_interest(pinf.get("field_of_interest"))
    if foi:
        inf["field_of_interest"] = foi

    if isinstance(pinf.get("gender"), str) and pinf["gender"].strip():
        inf["gender"] = pinf["gender"].strip()

    labels = _normalize_riasec_labels(pinf.get("riasec_attitudes"))
    if labels:
        inf["riasec_attitudes"] = labels

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


def clear_listener_patches() -> None:
    _LISTENER_PATCH_HISTORY.clear()
    _LISTENER_PATCH_BUFFER.clear()


def get_listener_patches() -> List[Dict[str, Any]]:
    return list(_LISTENER_PATCH_HISTORY)


def log_listener_patch(patch: Dict[str, Any]) -> None:
    if isinstance(patch, dict):
        _LISTENER_PATCH_HISTORY.append(copy.deepcopy(patch))


def strip_and_buffer_listener_patch(text: str) -> str:
    t = text or ""
    while True:
        m = _LISTENER_PATCH_BLOCK_RX.search(t)
        if not m:
            break
        raw_json = m.group(1)
        data = _safe_json_loads(raw_json)
        if isinstance(data, dict):
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
    expected_wrapup_riasec: str = ""
    advice_fallback: bool = False
    allowed_entities: set[str] = field(default_factory=set)
    allowed_locations: set[str] = field(default_factory=set)
    choice_options: List[str] = field(default_factory=list)
    advice_options: List[str] = field(default_factory=list)


_LAST_RAG_STATE = _RAGState()


def update_last_rag_state(
    ctx: str,
    *,
    query: str = "",
    phase: str = "",
    expected_verbatim: str = "",
    expected_wrapup_riasec: str = "",
    advice_fallback: bool = False,
    allowed_locations: Optional[Sequence[str]] = None,
    choice_options: Optional[Sequence[str]] = None,
    advice_options: Optional[Sequence[str]] = None,
) -> None:
    _LAST_RAG_STATE.ctx = ctx or ""
    _LAST_RAG_STATE.query = query or ""
    _LAST_RAG_STATE.phase = (phase or "").strip().lower()
    _LAST_RAG_STATE.expected_verbatim = (expected_verbatim or "").strip()
    _LAST_RAG_STATE.expected_wrapup_riasec = (expected_wrapup_riasec or "").strip().upper()
    _LAST_RAG_STATE.advice_fallback = bool(advice_fallback)

    _LAST_RAG_STATE.choice_options = [str(x).strip() for x in (choice_options or []) if str(x).strip()][:2]
    _LAST_RAG_STATE.advice_options = [
        str(x).strip()
        for x in (advice_options or [])
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

    _LAST_RAG_STATE.allowed_locations = set()
    if allowed_locations:
        for x in allowed_locations:
            nx = _normalize(str(x))
            if nx:
                _LAST_RAG_STATE.allowed_locations.add(nx)

def clear_last_rag_state() -> None:
    update_last_rag_state(
        "",
        query="",
        phase="",
        expected_verbatim="",
        expected_wrapup_riasec="",
        advice_fallback=False,
        allowed_locations=None,
        choice_options=None,
        advice_options=None,
    )


def get_last_rag_phase() -> str:
    return (_LAST_RAG_STATE.phase or "").strip().lower()


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
    expected_wrapup_riasec = (_LAST_RAG_STATE.expected_wrapup_riasec or "").strip().upper()

    yes_tok, no_tok = ("Sì", "No") if _DIALOG_LANGUAGE == "it" else ("Yes", "No")
    patch_block = _extract_listener_patch_block(ans)

    if phase == "qa_yesno":
        out = None
        if re.fullmatch(r"(?i)\s*(yes|sì|si)\s*", ans):
            out = yes_tok
        elif re.fullmatch(r"(?i)\s*no\s*", ans):
            out = no_tok
        else:
            cleaned = re.sub(r"[\.!\?\"\'\,]+", "", ans).strip()
            if re.fullmatch(r"(?i)\s*(yes|sì|si)\s*", cleaned):
                out = yes_tok
            elif re.fullmatch(r"(?i)\s*no\s*", cleaned):
                out = no_tok
            else:
                yes = bool(_FLOW_YES_RX.search(ans))
                no = bool(_FLOW_NO_RX.search(ans))
                out = yes_tok if (yes and not no) else (no_tok if (no and not yes) else no_tok)

        return (out + ("\n" + patch_block if patch_block else "")).strip()

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
        out = ans or _t("Goodbye.", "Arrivederci.")
        if patch_block:
            out = f"{out}\n{patch_block}"
        return out

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
    BG_QUESTIONS = [
        "What is your current education level (e.g., high school type, bachelor) and what subjects have you studied most recently?",
        "What subjects/topics do you enjoy most, and what are your strengths vs weaknesses? (Grades if you want to share.)",
        "What are your goals after graduation, and what constraints matter most (budget, language, mobility, preferred area in Italy: North, Center, South, or Islands)?",
    ]

    RIASEC_QUESTIONS = [
        "Do you like practical, hands-on work—building or repairing things?",
        "Are you curious to understand how things work?",
        "Do you enjoy researching, analyzing problems, and reasoning logically?",
        "Do you like expressing what you think or feel through creative forms (writing, art, music, design)?",
        "Do you like experimenting and creating new things?",
        "Do you like working closely with people and having a supportive/helping role?",
        "Do you like proposing ideas and organizing projects?",
        "Do you like having everything organized and under control?",
    ]

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

        default_bg_questions = list(self.BG_QUESTIONS)
        default_riasec_questions = list(self.RIASEC_QUESTIONS)

        self.BG_QUESTIONS = get_practice_list(
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
        self.retriever = retriever
        self.required_slots = tuple(required_slots)
        self.top_k = int(top_k)
        self.max_ctx_chars = int(max_ctx_chars)
        self.history_turns = int(history_turns)
        self.debug = bool(debug)
        self.min_options_target = max(1, int(min_options_target))

        self._slots = StudentSlots()
        self._listener_memory = _empty_listener_memory()

        self._flow_stage = "greet"
        self._bg_i = 0
        self._riasec_i = 0
        self._last_ctx: str = ""
        self._last_raw_blocks: List[str] = []
        self._advice_given = False
        self._pending_selection_ack = False

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
        self._last_raw_blocks = []
        self._advice_given = False
        self._pending_selection_ack = False

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

        mo = _OPT_RX.search(student_utt or "")
        if not mo:
            mnum = _NUM_ONLY_RX.search(student_utt or "")
            if mnum:
                mo = mnum

        if mo:
            n = int(mo.group(1))
            if self._last_options and 1 <= n <= len(self._last_options):
                selected = f"{n} - {self._last_options[n - 1]}"
            else:
                selected = str(n)

            self._slots.selected_option = selected

            exp = self._listener_memory.setdefault("explicit", {})
            if isinstance(exp, dict):
                exp["selected_option"] = selected

        self._sync_slots_from_listener_memory()
        record_debug_snapshot(self._listener_memory, self._slots.summary())

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
        mapping = {
            "cultural_heritage": [
                "cultural heritage",
                "patrimonio culturale",
                "beni culturali",
                "storia dell'arte",
                "archeologia",
            ],
            "arts_design": [
                "arts design",
                "art and design",
                "arti visive",
                "design",
                "arti",
            ],
            "design": [
                "design",
                "visual design",
                "grafica",
                "design della comunicazione",
            ],
            "communication_digital_media": [
                "digital communication",
                "digital media",
                "comunicazione digitale",
                "comunicazione e media",
            ],
            "humanities": [
                "humanities",
                "scienze umanistiche",
                "lettere",
                "studi umanistici",
            ],
            "psychology": [
                "psychology",
                "psicologia",
            ],
            "cognitive_science_linguistics": [
                "cognitive science",
                "linguistics",
                "mente e linguaggio",
                "scienze cognitive",
                "linguistica",
            ],
            "social_sciences": [
                "social sciences",
                "scienze sociali",
                "sociologia",
            ],
            "biology": [
                "biology",
                "biologia",
                "scienze biologiche",
            ],
            "environmental_science": [
                "environmental science",
                "scienze ambientali",
                "ambiente",
                "ecologia",
            ],
            "environmental_engineering": [
                "environmental engineering",
                "ingegneria ambientale",
                "ambiente e territorio",
            ],
            "computer_science": [
                "computer science",
                "informatica",
                "scienze informatiche",
            ],
        }
        return mapping.get(label, [label.replace("_", " ")])

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

            REGION_TO_RAG_MACROAREA = {
                "north": "Nord",
                "center": "Centro",
                "south": "Sud",
                "islands": "Isole",
            }

            if use_preferred_macroarea and self._slots.region:
                kwargs["preferred_macroarea"] = REGION_TO_RAG_MACROAREA.get(
                    self._slots.region,
                    self._slots.region,
                )

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

        if preferred_zone:
            zone_results: List[RetrievalScopeResult] = []

            chain = _build_university_chain(preferred_zone)

            for zone_name, allowed_universities in chain:
                raw_blocks = self._run_retrieval_scope(
                    queries,
                    allowed_universities=allowed_universities,
                    section_contains="Descrizione generale",
                    use_preferred_macroarea=False,
                )

                if not raw_blocks:
                    raw_blocks = self._run_retrieval_scope(
                        queries,
                        allowed_universities=allowed_universities,
                        section_contains=None,
                        use_preferred_macroarea=False,
                    )

                if not raw_blocks:
                    continue

                result = self._evaluate_scope_result(
                    scope_name=f"zone_{zone_name}",
                    raw_blocks=raw_blocks,
                )
                result.preferred_zone = preferred_zone
                result.matched_zone = zone_name

                # Importante: tieni solo le zone che producono opzioni vere.
                if result.options:
                    zone_results.append(result)

            if zone_results:
                merged_options = self._merge_zone_options_round_robin(zone_results)

                merged_blocks: List[str] = []
                for res in zone_results:
                    for blk in res.raw_blocks:
                        if blk not in merged_blocks:
                            merged_blocks.append(blk)

                aligned_blocks = self._select_blocks_for_options_in_order(
                    merged_blocks,
                    merged_options,
                    max_blocks=max(4, min(len(merged_options), self.candidate_pool_size)),
                )

                short_ctx = _build_short_ctx(
                    aligned_blocks,
                    self.max_ctx_chars,
                    max_blocks=min(len(merged_options), 8),
                )

                return RetrievalScopeResult(
                    scope_name="zoned_soft_merged",
                    raw_blocks=aligned_blocks,
                    short_ctx=short_ctx,
                    options=merged_options,
                    preferred_zone=preferred_zone,
                    matched_zone="mixed",
                )

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

        if not raw_blocks:
            return RetrievalScopeResult()

        result = self._evaluate_scope_result(
            scope_name="italy_fallback",
            raw_blocks=raw_blocks,
        )
        result.preferred_zone = preferred_zone
        result.matched_zone = None
        return result

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

        fields = [
            ("academic_background", clean(exp.get("academic_background"))),
            ("field_of_interest", foi_rendered),
            ("region", clean(exp.get("region"))),
            ("selected_option", clean(exp.get("selected_option"))),
            ("riasec_attitudes", riasec_rendered),
            ("gender", clean(inf.get("gender"))),
        ]

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

        if scope_name == "italy_fallback":
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
            return "STOP"

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
            update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="wrapup")

            riasec_labels = (self._listener_memory.get("inferred", {}) or {}).get("riasec_attitudes")
            code = _riasec_labels_to_code(riasec_labels) or "UNK"

            self._finished = True
            return (
                f"PHASE: WRAPUP\n"
                f"RIASEC: {code}\n"
                "Write the final message using EXACTLY this structure:\n"
                "Recap: <1-2 short sentences on background + fit>\n"
                f"RIASEC: {code}\n"
                "Next steps:\n"
                "1. <short concrete step>\n"
                "2. <short concrete step>\n"
                "3. <short concrete step>\n"
                "Goodbye: <short friendly goodbye>\n"
                "- Do NOT introduce new universities/programs.\n"
            )
        if self._advice_given and self._slots.selected_option and self._flow_stage != "qa":
            self._flow_stage = "qa"
            self._pending_selection_ack = ("?" not in (student_utt or ""))

        # ---------------------------------------------------------------------
        # Q&A
        # ---------------------------------------------------------------------
        if self._flow_stage == "qa":
            selection_patch = ""
            if (self._slots.selected_option or "") and (not self._choice_patch_requested):
                self._choice_patch_requested = True
                selection_patch = self._listener_patch_request(
                    targets="- explicit.selected_option",
                    rules=(
                        "- selected_option MUST be exactly in the format 'N - <EXACT option string from OPTIONS>'.\n"
                        "- Do NOT paraphrase.\n"
                        "- Use the exact option text shown in ADVICE."
                    ),
                    schema_example=(
                        '<LISTENER_PATCH>{"explicit":{"selected_option":'
                        '"1 - UNIVERSITÀ ..."}}'
                        "</LISTENER_PATCH>"
                    ),
                    evidence="Use the student's latest selection (Option 1/2/3) and the OPTIONS shown in ADVICE.",
                )

            def _ret(msg: str) -> str:
                return (msg + selection_patch) if selection_patch else msg

            if self._pending_selection_ack:
                self._pending_selection_ack = False
                update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_selected")
                return _ret(
                    "PHASE: Q&A\n"
                    f"The student selected: {self._slots.selected_option}\n"
                    "TASK:\n"
                    "- Acknowledge the selection in ONE short sentence.\n"
                    "- Do NOT introduce new universities/programs.\n"
                    "- Do NOT ask any questions.\n"
                )

            q_for_qa = self._extract_last_question_line(student_utt)
            qtype = self._classify_question(q_for_qa)

            if qtype == "choice":
                ch = self._extract_choice_options(q_for_qa)
                update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_choice", choice_options=ch)
                if len(ch) == 2:
                    return _ret(
                        "PHASE: Q&A\n"
                        + self._qa_context_block() +
                        "TASK:\n"
                        f"- Reply with ONLY ONE of these exact strings: '{ch[0]}' OR '{ch[1]}'.\n"
                        "- Output EXACTLY that string. Do NOT add any extra words.\n"
                    )
                return _ret(
                    "PHASE: Q&A\n"
                    + self._qa_context_block() +
                    "TASK:\n"
                    "- Reply with ONLY one of the two options mentioned in the student's question.\n"
                    "- Do NOT add any extra words.\n"
                )

            if qtype == "why":
                update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_why")
                return _ret(
                    "PHASE: Q&A\n"
                    + self._qa_context_block() +
                    "TASK:\n"
                    "- Answer the student's 'why' briefly.\n"
                    "- Ground your justification in the retrieved context; do not invent program details.\n"
                    "- Use the student's background + constraints + RIASEC to explain trade-offs.\n"
                    "- Do NOT introduce new universities/programs.\n"
                )

            if qtype == "yesno":
                update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_yesno")
                return _ret(
                    "PHASE: Q&A\n"
                    + self._qa_context_block() +
                    "TASK:\n"
                    "- Reply with ONLY 'Yes' or 'No'.\n"
                    "- Decide using the CONTEXT above (background, constraints, RIASEC, and the selected option).\n"
                )

            update_last_rag_state(self._last_ctx, query=_LAST_RAG_STATE.query, phase="qa_other")
            return _ret(
                "PHASE: Q&A\n"
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
                    "- Do not infer region from cities, towns, provinces, or vague descriptions.\n"
                    "\n"
                            "Rules for inferred.field_of_interest:\n"
                            "- Return field_of_interest as an ordered JSON array of 1 to 3 labels.\n"
                            "- Labels must be distinct and ordered from strongest to weakest fit.\n"
                            "- Choose labels only from this list:\n"
                            "  arts_design\n"
                            "  design\n"
                            "  communication_digital_media\n"
                            "  cultural_heritage\n"
                            "  humanities\n"
                            "  psychology\n"
                            "  cognitive_science_linguistics\n"
                            "  social_sciences\n"
                            "  biology\n"
                            "  environmental_science\n"
                            "  environmental_engineering\n"
                            "  computer_science\n"
                            "- Pick the intended university area, not the current job.\n"
                            "- Prefer concrete course-family areas over abstract hybrid concepts.\n"
                            "- Community art, visual expression, art practice -> arts_design.\n"
                            "- Art history, heritage, museums, cultural assets -> cultural_heritage.\n"
                            "- Biology/science + art/design -> biology or arts_design; prefer the side stated as the intended study direction.\n"
                            "- Tech + art/design/media -> communication_digital_media, design, or computer_science depending on emphasis.\n"
                            "- Writing + psychology + storytelling -> psychology, cognitive_science_linguistics, or humanities depending on emphasis.\n"
                            "- Environment/sustainability -> environmental_science; use environmental_engineering only if technical/engineering interest is explicit.\n"
                            "- If only one label is clearly supported, return a one-item array.\n"
                            "- If no label is clearly supported, omit the field.\n"
                            "- If two areas are both clearly supported by the student's message, return both labels.\n"
                            "\n"
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
                return (
                    "PHASE: ACQUIRE ACADEMIC BACKGROUND\n"
                    "Reply naturally in 1–2 short sentences, then ask ONE question (verbatim):\n"
                    f"{q}\n"
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
                need_academic_background = not (
                    isinstance(self._listener_memory, dict)
                    and isinstance(self._listener_memory.get("explicit"), dict)
                    and isinstance(self._listener_memory["explicit"].get("academic_background"), str)
                    and self._listener_memory["explicit"]["academic_background"].strip()
                )

                if self._riasec_i == 1:
                    bg_patch = self._listener_patch_request(
                        targets="- explicit.academic_background\n- inferred.field_of_interest\n- explicit.region",
                        rules=(
                            "- academic_background must summarize the student's current or most recent formal education. \n"
                            "- Prefer explicit study level and school/program name when available.\n"
                            "- Do NOT use jobs, hobbies, or goals as academic_background.\n"
                            "\n"
                            "Rules for inferred.field_of_interest:\n"
                            "- Return field_of_interest as an ordered JSON array of 1 to 3 labels.\n"
                            "- Labels must be distinct and ordered from strongest to weakest fit.\n"
                            "- Choose labels only from this list:\n"
                            "  arts_design\n"
                            "  design\n"
                            "  communication_digital_media\n"
                            "  cultural_heritage\n"
                            "  humanities\n"
                            "  psychology\n"
                            "  cognitive_science_linguistics\n"
                            "  social_sciences\n"
                            "  biology\n"
                            "  environmental_science\n"
                            "  environmental_engineering\n"
                            "  computer_science\n"
                            "- Pick the intended university area, not the current job.\n"
                            "- Prefer concrete course-family areas over abstract hybrid concepts.\n"
                            "- Community art, visual expression, art practice -> arts_design.\n"
                            "- Art history, heritage, museums, cultural assets -> cultural_heritage.\n"
                            "- Biology/science + art/design -> biology or arts_design; prefer the side stated as the intended study direction.\n"
                            "- Tech + art/design/media -> communication_digital_media, design, or computer_science depending on emphasis.\n"
                            "- Writing + psychology + storytelling -> psychology, cognitive_science_linguistics, or humanities depending on emphasis.\n"
                            "- Environment/sustainability -> environmental_science; use environmental_engineering only if technical/engineering interest is explicit.\n"
                            "- If only one label is clearly supported, return a one-item array.\n"
                            "- If no label is clearly supported, omit the field.\n"
                            "- If two areas are both clearly supported by the student's message, return both labels.\n"
                            "\n"
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
                    "Ask the following question VERBATIM. Ask ONE question only. Do NOT add anything else.\n"
                    f"{q}\n"
                    + bg_patch
                )

            self._flow_stage = "advice"

        # ---------------------------------------------------------------------
        # ADVICE
        # ---------------------------------------------------------------------
        query = self._build_query()
        retrieval, candidate_pool = self._get_llm_candidate_pool(query)

        self._last_ctx = retrieval.short_ctx
        self._last_raw_blocks = list(retrieval.raw_blocks)
        self._last_options = list(candidate_pool)

        riasec_patch = ""
        if not self._riasec_patch_requested:
            self._riasec_patch_requested = True
            riasec_patch = self._listener_patch_request(
                targets="- inferred.riasec_attitudes\n- inferred.gender",
                rules=(
                    "Use only the student's answers to the 8 RIASEC questions.\n"
                    "Return ONE JSON object inside <LISTENER_PATCH> ... </LISTENER_PATCH>.\n"
                    "Return only the requested fields.\n"
                    "Do not guess from the student's biography, job, or background.\n"
                    "\n"
                    "Allowed labels:\n"
                    "- Realistic\n"
                    "- Investigative\n"
                    "- Artistic\n"
                    "- Social\n"
                    "- Enterprising\n"
                    "- Conventional\n"
                    "\n"
                    "Map each question mainly to one label:\n"
                    "- practical, hands-on work -> Realistic\n"
                    "- curious to understand how things work -> Investigative\n"
                    "- researching, analyzing, logical reasoning -> Investigative\n"
                    "- creative expression -> Artistic\n"
                    "- experimenting and creating new things -> Artistic or Enterprising\n"
                    "- helping and supporting people -> Social\n"
                    "- proposing ideas and organizing projects -> Enterprising\n"
                    "- keeping everything organized and under control -> Conventional\n"
                    "\n"
                    "Rules:\n"
                    "- Return exactly 3 distinct labels, ordered from strongest to weakest.\n"
                    "- Positive answers support the matching label.\n"
                    "- Negative answers weaken that label.\n"
                    "- Do not include a label if the student clearly rejected it unless other answers strongly support it.\n"
                    "\n"
                    "Rules for inferred.gender:\n"
                    "- For gender, use the student's full conversation, especially the first message and any self-reference cues.\n"
                ),
                schema_example=(
                    '<LISTENER_PATCH>{"inferred":{"riasec_attitudes":["<top_trait>","<second_trait>","<third_trait>"],"gender":"<gender>"}}</LISTENER_PATCH>'
                ),
                evidence="Use the student's answers to the 8 RIASEC questions in this conversation.",
            )

        if retrieval.is_empty or not candidate_pool:
            self._advice_retry_count += 1

            if self._advice_retry_count <= 1:
                message, fallback_patch, should_finish = self._build_advice_fallback()

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
                "- Give exactly 3 concrete next steps, using only these ideas:\n"
                "  1. broaden the target field slightly,\n"
                "  2. keep the field exact and search again later when more universities are available,\n"
                "  3. review constraints/priorities.\n"
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
            + student_profile_block + "\n"
            + search_note
            + "CANDIDATE OPTIONS (already filtered for basic compatibility):\n"
            f"{opt_lines}\n\n"
            + "OPTION METADATA:\n"
            f"{option_meta}\n\n"
              "RAG_CONTEXT FOR THE CANDIDATE OPTIONS (paraphrase only; do not copy markers/IDs):\n"
              f"{retrieval.short_ctx}\n\n"
            + internal_riasec_step
            +"TASK:\n"
            "- Use ONLY the options listed in CANDIDATE OPTIONS.\n"
            "- These options are already filtered for basic compatibility, deduplicated, and cleaned.\n"
            "- The candidate options are not necessarily ordered. "
            "- Rank them using the student profile, including the inferred RIASEC labels.\n"
            "- Use the student profile to decide the ranking.\n"
            "- Prefer matches with the FIRST field_of_interest label over matches with later labels.\n"
            "- Region is a soft preference unless the student explicitly states they cannot relocate or must stay in one area.\n"
            "- Do NOT reject a stronger academic/profile fit only because it is outside the student's preferred area.\n"
            "- Prefer same-area options only when field fit, level fit, goals, and RIASEC fit are comparable.\n"
            "- Do NOT invent new programs, universities, course details, rankings, or explanations.\n"
            "- Return up to 3 options, only if they are genuinely plausible.\n"
            "- If no candidate is a plausible fit, state that no grounded option is available.\n"
            "\n"
            "OUTPUT RULES:\n"
            "- Output ONLY the final 3 options.\n"
            "- Use the exact option text from CANDIDATE OPTIONS.\n"
            "- Do not add explanations, comments, headers, bullets, or extra prose.\n"
            "- Output format must be EXACTLY:\n"
            "  1. <exact option text>\n"
            "  2. <exact option text>\n"
            "  3. <exact option text>\n"
            f'  Final line: "{final_line}"\n'
            + riasec_patch
        )

    def can_finish(self, utterance: str) -> bool:
        return self._finished


