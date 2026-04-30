# annotation_compact.py (CLEAN)
from __future__ import annotations

import re
from typing import List, Optional, Tuple

NOW_K = ("currently", "recently", "now", "at the moment", "these days", "right now", "today")
EDU_K = (
    "university", "college", "degree", "bachelor", "master", "phd", "major", "course",
    "study", "studying", "academic", "exam", "thesis", "research", "internship", "graduat",
)
CAREER_K = (
    "career", "job", "work", "working", "profession", "industry", "company", "role",
    "freelance", "consulting",
)
GOAL_K = (
    "i want", "i would like", "i'd like", "my goal", "i hope", "i'm hoping",
    "i plan", "i'm planning", "aspire", "dream", "aiming", "in the future",
)
CONSTRAINT_K = (
    "budget", "money", "cost", "tuition", "scholar", "visa", "deadline", "time", "family",
    "language", "english", "italian", "relocat", "move", "commute",
    "stress", "anx", "worried", "worry", "afraid", "fear", "pressure", "mental health",
)
PREF_K = (
    "prefer", "i prefer", "i like", "i enjoy", "interested", "interest", "passion",
    "curious", "motivated", "value",
)

_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

def _split_sentences(text: str) -> List[str]:
    txt = _WS_RE.sub(" ", (text or "").strip())
    if not txt:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(txt) if s.strip()]

def _score(s: str, keywords: Tuple[str, ...]) -> int:
    sl = s.lower()
    return sum(sl.count(k) for k in keywords)

def _best(sents: List[str], keywords: Tuple[str, ...], chosen: List[str], min_score: int = 1) -> Optional[str]:
    best_s, best_sc = None, 0
    chosen_set = {c.lower() for c in chosen}
    for s in sents:
        if s.lower() in chosen_set:
            continue
        sc = _score(s, keywords)
        if sc < min_score:
            continue
        if sc > best_sc or (sc == best_sc and (best_s is None or len(s) < len(best_s))):
            best_s, best_sc = s, sc
    return best_s

def _trim(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    if max_chars <= 1:
        return "…"
    cut = max_chars - 1
    chunk = t[:cut]
    # prova a tagliare su spazio, non troppo presto
    last_space = chunk.rfind(" ")
    if last_space >= int(cut * 0.6):
        chunk = chunk[:last_space]
    return chunk.rstrip() + "…"

def compact_annotation(annotation: str, max_chars: int = 550) -> str:
    """
    Riduce l'annotation in 3–4 bullet utili per prompt:
    - Background (prima frase)
    - Studio/lavoro attuale
    - Obiettivo
    - Vincoli / preferenze
    """
    sents = _split_sentences(annotation or "")
    if not sents:
        return ""

    background = sents[0]
    chosen = [background]

    study = _best(sents, NOW_K + EDU_K + CAREER_K, chosen, min_score=1)
    if study:
        chosen.append(study)

    goal = _best(sents, GOAL_K, chosen, min_score=1)
    if goal:
        chosen.append(goal)

    cp = _best(sents, CONSTRAINT_K, chosen, min_score=1) or _best(sents, PREF_K, chosen, min_score=1)
    if cp:
        chosen.append(cp)

    bullets = []
    bullets.append(f"- Background: {background}")
    if study:
        bullets.append(f"- Study/Career (now): {study}")
    if goal:
        bullets.append(f"- Goal: {goal}")
    if cp:
        bullets.append(f"- Constraints/Preferences: {cp}")

    out = "\n".join(bullets).strip()
    if len(out) > max_chars:
        out = _trim(out.replace("\n", " "), max_chars)
    return out