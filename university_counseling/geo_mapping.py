from __future__ import annotations

import re
import hashlib
from typing import Optional, Dict, List, Tuple

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
# (aggiungi pure altre città se vuoi)
# -------------------------
CITY_TO_REGION: Dict[str, str] = {
    "aosta": "Valle d'Aosta",
    "torino": "Piemonte",
    "genova": "Liguria",
    "milano": "Lombardia",
    "brescia": "Lombardia",
    "bergamo": "Lombardia",
    "trento": "Trentino-Alto Adige",
    "bolzano": "Trentino-Alto Adige",
    "venezia": "Veneto",
    "verona": "Veneto",
    "padova": "Veneto",
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
    "campobasso": "Molise",
    "napoli": "Campania",
    "naples": "Campania",
    "salerno": "Campania",
    "bari": "Puglia",
    "lecce": "Puglia",
    "potenza": "Basilicata",
    "catanzaro": "Calabria",
    "reggio calabria": "Calabria",
    "palermo": "Sicilia",
    "catania": "Sicilia",
    "cagliari": "Sardegna",
    "sassari": "Sardegna",
}

# -------------------------
# Foreign / vague -> macro
# (qui inserisci i tuoi mapping “bias-related”)
# -------------------------
FOREIGN_TO_MACRO: Dict[str, str] = {
    "midwest": "Nord",
    "north east": "Nord",
    "northeast": "Nord",
    "east coast": "Centro",
    "west coast": "Centro",
    "south": "Sud",      # generico: US South ecc.
    "southern": "Sud",
}

# Varianti macro-italiane nel testo
MACRO_ALIASES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(north(ern)?\s+italy|nord(\s+italia)?)\b", re.I), "Nord"),
    (re.compile(r"\b(central\s+italy|centro(\s+italia)?)\b", re.I), "Centro"),
    (re.compile(r"\b(south(ern)?\s+italy|sud(\s+italia)?)\b", re.I), "Sud"),
    (re.compile(r"\b(islands|isole)\b", re.I), "Isole"),
]

def _stable_pick(seed: str, items: List[str]) -> str:
    h = hashlib.md5(seed.encode("utf-8", errors="ignore")).hexdigest()
    return items[int(h[:8], 16) % len(items)]

def normalize_location_to_region(text: str, *, seed: str = "0") -> Optional[str]:
    """
    Ritorna una REGIONE italiana canonica (es. 'Toscana').
    Se trova solo una macro-area (Nord/Centro/Sud/Isole), sceglie una regione coerente in modo deterministico.
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    # 1) città -> regione
    for city, region in CITY_TO_REGION.items():
        if re.search(rf"\b{re.escape(city)}\b", t):
            return region

    # 2) macro area esplicita italiana
    for rx, macro in MACRO_ALIASES:
        if rx.search(t):
            regs = MACRO_TO_REGIONS.get(macro) or []
            return _stable_pick(f"{seed}|{macro}", regs) if regs else None

    # 3) foreign keyword -> macro -> regione
    for key, macro in FOREIGN_TO_MACRO.items():
        if re.search(rf"\b{re.escape(key)}\b", t):
            regs = MACRO_TO_REGIONS.get(macro) or []
            return _stable_pick(f"{seed}|{key}|{macro}", regs) if regs else None

    return None

def region_to_macro(region: str) -> Optional[str]:
    return ITALY_REGION_TO_MACRO.get(region)