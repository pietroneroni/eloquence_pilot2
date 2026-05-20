# lancedb_university_retriever.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer


def _norm(s: str) -> str:
    return (s or "").casefold().strip()


def _norm_macro(s: str) -> str:
    x = _norm(s)
    aliases = {
        "north": "nord",
        "northern italy": "nord",
        "center": "centro",
        "centre": "centro",
        "central italy": "centro",
        "south": "sud",
        "southern italy": "sud",
        "islands": "isole",
        "the italian islands": "isole",
    }
    return aliases.get(x, x)


class LanceDBUniversityRetriever:
    def __init__(
        self,
        *,
        lancedb_dir: str,
        table_name: str = "universities",
        embed_model: str = "intfloat/multilingual-e5-base",
    ) -> None:
        self.db = lancedb.connect(lancedb_dir)
        self.table = self.db.open_table(table_name)
        self.model = SentenceTransformer(embed_model)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        university: str | None = None,
        course_contains: str | None = None,
        section_contains: str | None = None,
        preferred_region: str | None = None,
        preferred_macroarea: str | None = None,
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
        pref_region = _norm(preferred_region) if preferred_region else ""
        pref_macro = _norm_macro(preferred_macroarea) if preferred_macroarea else ""

        hits: list[dict[str, Any]] = []

        for rank_index, row in enumerate(rows):
            raw_meta = row.get("metadata") or "{}"
            try:
                meta = json.loads(raw_meta)
            except Exception:
                meta = {}

            text = row.get("text") or ""

            university_name = _norm(str(meta.get("UNIVERSITY", "")))
            course_name = _norm(str(meta.get("COURSE", "")))
            course_code = _norm(str(meta.get("COURSE_CODE", "")))
            section_name = _norm(str(meta.get("SECTION", "")))
            hit_region = _norm(str(meta.get("REGION", "")))
            hit_macro = _norm_macro(str(meta.get("MACROAREA", "")))

            if uni_f and uni_f not in university_name:
                continue

            if course_f and course_f not in course_name and course_f not in course_code:
                continue

            if section_f and section_f not in section_name:
                continue

            distance = float(row.get("_distance", 0.0))
            base_score = -distance

            bonus = 0.0
            if pref_region and hit_region == pref_region:
                bonus += 0.15
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
            course_key = _norm(str(meta.get("COURSE_CODE") or meta.get("COURSE") or ""))

            if not course_key:
                course_key = _norm(str(h.get("text") or "")[:120])

            if course_key in seen_courses:
                continue

            seen_courses.add(course_key)
            deduped.append(h)

            if len(deduped) >= top_k:
                break

        return deduped