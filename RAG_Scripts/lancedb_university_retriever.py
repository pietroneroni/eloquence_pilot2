from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_DATASET_DIRS = {
    "eng": "RAG_University_Eng",
    "ita": "RAG_University_Ita",
}
SUPPORTED_LANGS = ("eng", "ita")


def _norm(s: str) -> str:
    return (s or "").casefold().strip()


def _ascii_norm(s: str) -> str:
    t = unicodedata.normalize("NFKD", s or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.casefold()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _norm_macro(s: str) -> str:
    x = _ascii_norm(s)
    aliases = {
        "north": "nord",
        "northern italy": "nord",
        "nord": "nord",
        "nord italia": "nord",
        "italia del nord": "nord",
        "center": "centro",
        "centre": "centro",
        "central italy": "centro",
        "centro": "centro",
        "centro italia": "centro",
        "italia centrale": "centro",
        "south": "sud",
        "southern italy": "sud",
        "sud": "sud",
        "sud italia": "sud",
        "italia del sud": "sud",
        "islands": "isole",
        "the italian islands": "isole",
        "italian islands": "isole",
        "isole": "isole",
        "isole italiane": "isole",
    }
    return aliases.get(x, x)


def _normalize_language(lang: str | None) -> str:
    lang = (lang or "eng").strip().lower()
    aliases = {
        "en": "eng",
        "english": "eng",
        "inglese": "eng",
        "it": "ita",
        "italian": "ita",
        "italiano": "ita",
    }
    lang = aliases.get(lang, lang)
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language '{lang}'. Use one of: {', '.join(SUPPORTED_LANGS)}")
    return lang


def _default_rag_dir(project_root: Path, lang: str) -> Path:
    lang = _normalize_language(lang)
    preferred = project_root / DEFAULT_DATASET_DIRS[lang]

    # Backward compatibility with old layout.
    if lang == "eng" and not preferred.exists():
        legacy = project_root / "RAG_University"
        if legacy.exists():
            return legacy

    return preferred


def _keyword_in_text(text: str, keyword: str) -> bool:
    t = _ascii_norm(text)
    k = _ascii_norm(keyword)
    if not t or not k:
        return False
    if " " in k:
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", t))
    toks = set(t.split())
    variants = {k}
    if k.endswith("y") and len(k) > 2:
        variants.add(k[:-1] + "ies")
    else:
        variants.add(k + "s")
        variants.add(k + "es")
    return any(v in toks for v in variants)


def _university_allowed(university: str, allowed: list[str] | None) -> bool:
    if not allowed:
        return True
    u = _norm(university)
    for a in allowed:
        aa = _norm(a)
        if aa and (u == aa or u in aa or aa in u):
            return True
    return False


def _metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_meta = row.get("metadata") or row.get("meta") or "{}"
    if isinstance(raw_meta, dict):
        return raw_meta
    try:
        return json.loads(raw_meta)
    except Exception:
        return {}


class LanceDBUniversityRetriever:
    def __init__(
        self,
        *,
        lang: str = "eng",
        rag_dir: str | None = None,
        lancedb_dir: str | None = None,
        table_name: str = "universities",
        embed_model: str = "intfloat/multilingual-e5-base",
    ) -> None:
        self.lang = _normalize_language(lang)
        project_root = Path(__file__).resolve().parents[1]

        if not lancedb_dir:
            dataset_dir = Path(rag_dir) if rag_dir else _default_rag_dir(project_root, self.lang)
            lancedb_dir = str(dataset_dir / "Embeddings" / "lancedb")

        self.lancedb_dir = lancedb_dir
        self.db = lancedb.connect(lancedb_dir)
        self.table = self.db.open_table(table_name)
        self.model = SentenceTransformer(embed_model)
        self._all_rows_cache: list[dict[str, Any]] | None = None

    def _all_rows(self) -> list[dict[str, Any]]:
        if self._all_rows_cache is not None:
            return self._all_rows_cache
        try:
            rows = self.table.to_list()
        except Exception:
            try:
                rows = self.table.to_pandas().to_dict("records")
            except Exception:
                rows = []
        self._all_rows_cache = list(rows or [])
        return self._all_rows_cache

    def keyword_search(
        self,
        *,
        keywords: list[str],
        top_k: int = 20,
        preferred_region: str | None = None,
        strict_region: bool = False,
        preferred_macroarea: str | None = None,
        strict_macroarea: bool = False,
        allowed_universities: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        terms = [str(k).strip() for k in (keywords or []) if str(k).strip()]
        if not terms:
            return []

        pref_region = _ascii_norm(preferred_region) if preferred_region else ""
        pref_macro = _norm_macro(preferred_macroarea) if preferred_macroarea else ""
        scored: list[dict[str, Any]] = []

        for rank_index, row in enumerate(self._all_rows()):
            meta = _metadata_from_row(row)
            text = str(row.get("text") or "")
            university_name = str(meta.get("UNIVERSITY", ""))
            course_name = str(meta.get("COURSE", ""))
            course_code = str(meta.get("COURSE_CODE", ""))
            section_name = str(meta.get("SECTION", ""))
            hit_region = _ascii_norm(str(meta.get("REGION", "")))
            hit_macro = _norm_macro(str(meta.get("MACROAREA", "")))

            if strict_region and pref_region and hit_region != pref_region:
                continue
            if strict_macroarea and pref_macro and hit_macro != pref_macro:
                continue
            if not _university_allowed(university_name, allowed_universities):
                continue

            title_blob = " ".join([course_name, course_code])
            full_blob = " ".join([university_name, course_name, course_code, section_name, text])

            title_hits = sum(1 for term in terms if _keyword_in_text(title_blob, term))
            full_hits = sum(1 for term in terms if _keyword_in_text(full_blob, term))
            if full_hits <= 0:
                continue

            score = title_hits * 3.0 + full_hits * 1.0
            if pref_region and hit_region == pref_region:
                score += 0.8
            if pref_macro and hit_macro == pref_macro:
                score += 0.5

            scored.append({
                "score": score,
                "base_score": score,
                "text": text,
                "meta": meta,
                "id": row.get("id"),
                "rank_index": rank_index,
            })

        scored.sort(key=lambda x: (x["score"], -int(x.get("rank_index", 0))), reverse=True)

        deduped: list[dict[str, Any]] = []
        seen_courses = set()
        for h in scored:
            meta = h.get("meta") or {}
            university_key = _norm(str(meta.get("UNIVERSITY") or ""))
            course_key = _norm(str(meta.get("COURSE_CODE") or meta.get("COURSE") or ""))
            key = f"{university_key}|{course_key}" if course_key else _norm(str(h.get("text") or "")[:120])
            if key in seen_courses:
                continue
            seen_courses.add(key)
            deduped.append(h)
            if len(deduped) >= top_k:
                break
        return deduped

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        university: str | None = None,
        course_contains: str | None = None,
        section_contains: str | None = None,
        preferred_region: str | None = None,
        strict_region: bool = False,
        preferred_macroarea: str | None = None,
        strict_macroarea: bool = False,
        dedupe_by_course: bool = True,
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        query_vec = self.model.encode(
            [f"query: {q}"],
            normalize_embeddings=True,
        )
        query_vec = np.asarray(query_vec[0], dtype="float32")

        fetch_k = max(top_k * 30, 100)
        rows = self.table.search(query_vec, vector_column_name="vector").limit(fetch_k).to_list()

        uni_f = _norm(university) if university else ""
        course_f = _norm(course_contains) if course_contains else ""
        section_f = _norm(section_contains) if section_contains else ""
        pref_region = _ascii_norm(preferred_region) if preferred_region else ""
        pref_macro = _norm_macro(preferred_macroarea) if preferred_macroarea else ""

        hits: list[dict[str, Any]] = []

        for rank_index, row in enumerate(rows):
            meta = _metadata_from_row(row)
            text = row.get("text") or ""

            university_name = _norm(str(meta.get("UNIVERSITY", "")))
            course_name = _norm(str(meta.get("COURSE", "")))
            course_code = _norm(str(meta.get("COURSE_CODE", "")))
            section_name = _norm(str(meta.get("SECTION", "")))
            hit_region = _ascii_norm(str(meta.get("REGION", "")))
            hit_macro = _norm_macro(str(meta.get("MACROAREA", "")))

            if uni_f and uni_f not in university_name:
                continue

            if course_f and course_f not in course_name and course_f not in course_code:
                continue

            if section_f and section_f not in section_name:
                continue

            if strict_region and pref_region and hit_region != pref_region:
                continue

            if strict_macroarea and pref_macro and hit_macro != pref_macro:
                continue

            distance = float(row.get("_distance", 0.0))
            base_score = -distance

            bonus = 0.0
            if pref_region and hit_region == pref_region:
                bonus += 0.20
            if pref_macro and hit_macro == pref_macro:
                bonus += 0.10

            hits.append({
                "score": base_score + bonus,
                "base_score": base_score,
                "text": text,
                "meta": meta,
                "id": row.get("id"),
                "rank_index": rank_index,
            })

        hits.sort(key=lambda x: x["score"], reverse=True)

        if not dedupe_by_course:
            return hits[:top_k]

        deduped = []
        seen_courses = set()

        for h in hits:
            meta = h.get("meta") or {}
            university_key = _norm(str(meta.get("UNIVERSITY") or ""))
            course_key = _norm(str(meta.get("COURSE_CODE") or meta.get("COURSE") or ""))

            if course_key:
                course_key = f"{university_key}|{course_key}"
            else:
                course_key = _norm(str(h.get("text") or "")[:120])

            if course_key in seen_courses:
                continue

            seen_courses.add(course_key)
            deduped.append(h)

            if len(deduped) >= top_k:
                break

        return deduped
