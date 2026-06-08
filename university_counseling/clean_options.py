
import re
import unicodedata
from difflib import SequenceMatcher


NOISE_TOKENS = {
    "warning_amber_outline",
    "info",
    "salva pdf",
    "cambia",
}

UI_NOISE_PATTERNS = [
    r"\bwarning_amber_outline\b",
    r"\binfo insegnamenti(?: erogati)?\b",
    r"\bsalva pdf\b",
    r"\bcambia\b",
    r"\bpercorso comune\b",
]

COURSE_CODE_RE = re.compile(r"\[([A-Z0-9]+)\]")
MULTISPACE_RE = re.compile(r"\s+")
OPTION_PREFIX_RE = re.compile(r"^\s*(?:-\s*)?(?:Option\s+\d+:\s*)?", re.IGNORECASE)


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _normalize_spaces(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def _clean_option_text(text: str) -> str:
    text = OPTION_PREFIX_RE.sub("", text or "")
    text = text.replace("–", "-").replace("—", "-")
    for pat in UI_NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = _normalize_spaces(text)
    return text


def _looks_valid_option(text: str) -> bool:
    """
    Filtro conservativo:
    - deve avere separatore university | course
    - deve avere course code [....]
    - deve avere abbastanza testo reale
    """
    if not text:
        return False

    if "|" not in text:
        return False

    if not COURSE_CODE_RE.search(text):
        return False

    if len(text) < 20:
        return False

    lowered = _strip_accents(text.lower())
    noise_hits = sum(tok in lowered for tok in NOISE_TOKENS)
    if noise_hits >= 2:
        return False

    return True


def _parse_option(text: str) -> dict:
    """
    Atteso formato:
    'Università ... | [1234] TITOLO CORSO ...'
    """
    left, right = text.split("|", 1)
    university = _normalize_spaces(left)
    right = _normalize_spaces(right)

    code_match = COURSE_CODE_RE.search(right)
    code = code_match.group(1).upper() if code_match else None

    title = right
    if code_match:
        title = right[code_match.end():].strip()

    return {
        "raw": text,
        "university": university,
        "code": code,
        "title": title,
    }


def _normalized_title_for_similarity(title: str) -> str:
    t = _strip_accents(title.lower())
    t = re.sub(r"\b(curriculum|percorso|track|indirizzo|interateneo)\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = _normalize_spaces(t)
    return t


def _informativeness_score(parsed: dict) -> tuple:
    """
    Più alto = meglio.
    Scelta pratica:
    - preferisci righe pulite
    - preferisci titolo informativo ma non eccessivamente rumoroso
    - se esiste una specializzazione utile, in genere meglio del titolo troppo generico
    """
    title = parsed["title"]
    norm_title = _normalized_title_for_similarity(title)

    has_curriculum_hint = bool(re.search(
        r"\b(arch(e|eo)ologia|storia|documentazione|beni|gestione|clinica|salute|well-being|performance|comunicazione|digitali|ecosistemi|ecologico|marino)\b",
        norm_title,
        flags=re.IGNORECASE
    ))

    score = 0
    score += min(len(norm_title), 120)
    score += 20 if has_curriculum_hint else 0
    score -= 15 if "warning_amber_outline" in parsed["raw"].lower() else 0
    return (score, len(parsed["raw"]))


def _similar(a: str, b: str, threshold: float = 0.88) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold


def clean_and_dedup_options(raw_options: list[str], max_options: int | None = None) -> list[str]:
    """
    Pipeline:
    1) clean text
    2) drop malformed/noisy rows
    3) dedup exact family by (university, code)
    4) secondary near-dup pass by same university + very similar title
    """
    cleaned = []
    for opt in raw_options:
        txt = _clean_option_text(opt)
        if not _looks_valid_option(txt):
            continue
        cleaned.append(txt)

    # 1) dedup by (university, code)
    best_by_family: dict[tuple[str, str], dict] = {}
    for txt in cleaned:
        parsed = _parse_option(txt)
        fam_key = (
            _strip_accents(parsed["university"].lower()),
            (parsed["code"] or "").upper(),
        )

        current_best = best_by_family.get(fam_key)
        if current_best is None:
            best_by_family[fam_key] = parsed
            continue

        if _informativeness_score(parsed) > _informativeness_score(current_best):
            best_by_family[fam_key] = parsed

    deduped = list(best_by_family.values())

    # 2) secondary near-duplicate pass:
    # stessa università + titolo quasi uguale -> tieni il migliore, ma preserva
    # l'ordine di retrieval. Ordinare per informativeness qui altera ranking e
    # vincoli geografici già decisi a monte.
    final_items: list[dict] = []
    for cand in deduped:
        cand_uni = _strip_accents(cand["university"].lower())
        cand_title = _normalized_title_for_similarity(cand["title"])

        dup_index = None
        for idx, kept in enumerate(final_items):
            kept_uni = _strip_accents(kept["university"].lower())
            kept_title = _normalized_title_for_similarity(kept["title"])
            if cand_uni == kept_uni and _similar(cand_title, kept_title):
                dup_index = idx
                break

        if dup_index is None:
            final_items.append(cand)
        elif _informativeness_score(cand) > _informativeness_score(final_items[dup_index]):
            final_items[dup_index] = cand

    results = [item["raw"] for item in final_items]

    if max_options is not None:
        results = results[:max_options]

    return results
