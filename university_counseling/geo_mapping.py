from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# -------------------------
# Canonical regions + macro
# -------------------------
ITALY_REGION_TO_MACRO: Dict[str, str] = {
    "Valle d'Aosta": "Nord",
    "Piemonte": "Nord",
    "Liguria": "Nord",
    "Lombardia": "Nord",
    "Trentino-Alto Adige": "Nord",
    "Veneto": "Nord",
    "Friuli-Venezia Giulia": "Nord",
    "Emilia-Romagna": "Nord",
    "Toscana": "Centro",
    "Umbria": "Centro",
    "Marche": "Centro",
    "Lazio": "Centro",
    "Abruzzo": "Sud",
    "Molise": "Sud",
    "Campania": "Sud",
    "Puglia": "Sud",
    "Basilicata": "Sud",
    "Calabria": "Sud",
    "Sicilia": "Isole",
    "Sardegna": "Isole",
}

MACRO_TO_REGIONS: Dict[str, List[str]] = {}
for reg, macro in ITALY_REGION_TO_MACRO.items():
    MACRO_TO_REGIONS.setdefault(macro, []).append(reg)

# -------------------------
# City -> region
# -------------------------
CITY_TO_REGION: Dict[str, str] = {
    "aosta": "Valle d'Aosta",
    "torino": "Piemonte",
    "turin": "Piemonte",
    "genova": "Liguria",
    "genoa": "Liguria",
    "milano": "Lombardia",
    "milan": "Lombardia",
    "brescia": "Lombardia",
    "bergamo": "Lombardia",
    "trento": "Trentino-Alto Adige",
    "bolzano": "Trentino-Alto Adige",
    "venezia": "Veneto",
    "venice": "Veneto",
    "verona": "Veneto",
    "padova": "Veneto",
    "padua": "Veneto",
    "trieste": "Friuli-Venezia Giulia",
    "udine": "Friuli-Venezia Giulia",
    "bologna": "Emilia-Romagna",
    "parma": "Emilia-Romagna",
    "modena": "Emilia-Romagna",
    "firenze": "Toscana",
    "florence": "Toscana",
    "pisa": "Toscana",
    "siena": "Toscana",
    "perugia": "Umbria",
    "ancona": "Marche",
    "roma": "Lazio",
    "rome": "Lazio",
    "l'aquila": "Abruzzo",
    "laquila": "Abruzzo",
    "campobasso": "Molise",
    "napoli": "Campania",
    "naples": "Campania",
    "salerno": "Campania",
    "bari": "Puglia",
    "lecce": "Puglia",
    "foggia": "Puglia",
    "potenza": "Basilicata",
    "catanzaro": "Calabria",
    "reggio calabria": "Calabria",
    "palermo": "Sicilia",
    "catania": "Sicilia",
    "messina": "Sicilia",
    "cagliari": "Sardegna",
    "sassari": "Sardegna",
}

# Important: no foreign/vague geography is mapped into an Italian macro-area.
# Terms like "Midwest", "East Coast" or a bare "South" outside an Italian-area
# answer are intentionally ambiguous. The counselor should ask for clarification
# rather than silently projecting them onto Nord/Centro/Sud/Isole.
FOREIGN_OR_AMBIGUOUS_LOCATION_TERMS = {
    "midwest",
    "mid-west",
    "northeast",
    "north-east",
    "north east",
    "east coast",
    "west coast",
    "usa",
    "united states",
    "america",
    "south",
    "southern",
}

MACRO_ALIASES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(north(ern)?\s+italy|nord(\s+italia)?|italia\s+del\s+nord)\b", re.I), "Nord"),
    (re.compile(r"\b(central\s+italy|centro(\s+italia)?|italia\s+centrale)\b", re.I), "Centro"),
    (re.compile(r"\b(south(ern)?\s+italy|sud(\s+italia)?|italia\s+del\s+sud)\b", re.I), "Sud"),
    (re.compile(r"\b(islands|italian\s+islands|isole|isole\s+italiane|sicilia|sardegna)\b", re.I), "Isole"),
]

# Bare macro aliases are safe only when the caller knows the utterance is an
# answer to an Italian-area question such as "Nord, Centro, Sud o Isole?".
BARE_MACRO_ALIASES: Dict[str, str] = {
    "north": "Nord",
    "nord": "Nord",
    "center": "Centro",
    "centre": "Centro",
    "centro": "Centro",
    "south": "Sud",
    "sud": "Sud",
    "islands": "Isole",
    "isole": "Isole",
}


def _strip_accents(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _norm(text: str) -> str:
    t = _strip_accents(text).casefold()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _stable_pick(seed: str, items: List[str]) -> str:
    if not items:
        return ""
    h = hashlib.md5(seed.encode("utf-8", errors="ignore")).hexdigest()
    return items[int(h[:8], 16) % len(items)]


_REGION_NORM_TO_CANONICAL = {_norm(region): region for region in ITALY_REGION_TO_MACRO}


def normalize_italian_region_name(text: str) -> Optional[str]:
    """Return a canonical Italian region if text names one, else None."""
    t = _norm(text)
    if not t:
        return None
    if t in _REGION_NORM_TO_CANONICAL:
        return _REGION_NORM_TO_CANONICAL[t]

    # Common variants that do not normalize one-to-one.
    variants = {
        "valle d aosta": "Valle d'Aosta",
        "val d aosta": "Valle d'Aosta",
        "trentino alto adige": "Trentino-Alto Adige",
        "friuli venezia giulia": "Friuli-Venezia Giulia",
        "emilia romagna": "Emilia-Romagna",
    }
    return variants.get(t)


def normalize_location_to_region(
    text: str,
    *,
    seed: str = "0",
    allow_city_inference: bool = True,
    allow_bare_macro: bool = False,
) -> Optional[str]:
    """Normalize an Italian location mention to a canonical Italian region.

    The function deliberately does not map foreign or vague geography to Italy.
    For example, "Midwest" and "East Coast" return None. A bare "South" also
    returns None unless allow_bare_macro=True, because outside the explicit
    Italian-area question it can mean many things.
    """
    raw = (text or "").strip()
    t = _norm(raw)
    if not t:
        return None

    direct_region = normalize_italian_region_name(raw)
    if direct_region:
        return direct_region

    if allow_city_inference:
        # Prefer longer city names first so "reggio calabria" wins over
        # possible shorter substrings.
        for city, region in sorted(CITY_TO_REGION.items(), key=lambda kv: len(kv[0]), reverse=True):
            city_norm = _norm(city)
            if re.search(rf"(?<![a-z0-9]){re.escape(city_norm)}(?![a-z0-9])", t):
                return region

    for rx, macro in MACRO_ALIASES:
        if rx.search(raw):
            regs = MACRO_TO_REGIONS.get(macro) or []
            return _stable_pick(f"{seed}|{macro}", regs) if regs else None

    if allow_bare_macro:
        macro = BARE_MACRO_ALIASES.get(t)
        if macro:
            regs = MACRO_TO_REGIONS.get(macro) or []
            return _stable_pick(f"{seed}|{macro}", regs) if regs else None

    return None


def location_is_ambiguous_or_foreign(text: str) -> bool:
    t = _norm(text)
    if not t:
        return False
    for term in FOREIGN_OR_AMBIGUOUS_LOCATION_TERMS:
        term_norm = _norm(term)
        if re.search(rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])", t):
            return True
    return False


def region_to_macro(region: str) -> Optional[str]:
    canonical = normalize_italian_region_name(region) or region
    return ITALY_REGION_TO_MACRO.get(canonical)
