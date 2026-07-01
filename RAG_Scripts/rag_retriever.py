from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import faiss
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
        "center": "centro",
        "centre": "centro",
        "central italy": "centro",
        "centro": "centro",
        "south": "sud",
        "southern italy": "sud",
        "sud": "sud",
        "islands": "isole",
        "the italian islands": "isole",
        "italian islands": "isole",
        "isole": "isole",
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


class RAGRetriever:
    def __init__(
        self,
        *,
        lang: str = "eng",
        rag_dir: str | None = None,
        embeddings_dir: str | None = None,
        faiss_path: str | None = None,
        chunks_path: str | None = None,
        embed_model: str = "intfloat/multilingual-e5-base",
    ) -> None:
        self.lang = _normalize_language(lang)
        project_root = Path(__file__).resolve().parents[1]

        if embeddings_dir:
            base = Path(embeddings_dir)
        else:
            dataset_dir = Path(rag_dir) if rag_dir else _default_rag_dir(project_root, self.lang)
            base = dataset_dir / "Embeddings"

        faiss_path = faiss_path or str(base / "rag.index.faiss")
        chunks_path = chunks_path or str(base / "rag.chunks.jsonl")

        self.faiss_path = faiss_path
        self.chunks_path = chunks_path
        self.model = SentenceTransformer(embed_model)

        fp = Path(self.faiss_path)
        cp = Path(self.chunks_path)

        if not fp.exists():
            raise FileNotFoundError(f"FAISS index not found: {fp.resolve()} (cwd={Path.cwd()})")
        if not cp.exists():
            raise FileNotFoundError(f"Chunks file not found: {cp.resolve()} (cwd={Path.cwd()})")

        self.index = faiss.read_index(str(fp))

        self.chunks: list[dict[str, Any]] = []
        with open(cp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    item.setdefault("meta", {}).setdefault("LANGUAGE", self.lang)
                    self.chunks.append(item)

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

        q = f"query: {q}"

        uni_f = _norm(university) if university else ""
        course_f = _norm(course_contains) if course_contains else ""
        section_f = _norm(section_contains) if section_contains else ""
        pref_region = _ascii_norm(preferred_region) if preferred_region else ""
        pref_macro = _norm_macro(preferred_macroarea) if preferred_macroarea else ""

        emb = self.model.encode([q], normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")

        fetch_k = min(max(top_k * 30, 100), len(self.chunks))
        scores, idx = self.index.search(emb, fetch_k)

        hits: list[dict[str, Any]] = []

        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue

            item = self.chunks[int(i)]
            meta = item.get("meta") or {}
            text = item.get("text") or ""

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

            bonus = 0.0

            if pref_region and hit_region == pref_region:
                bonus += 0.20

            if pref_macro and hit_macro == pref_macro:
                bonus += 0.10

            hits.append(
                {
                    "score": float(s) + bonus,
                    "base_score": float(s),
                    "text": text,
                    "meta": meta,
                    "id": item.get("id"),
                    "rank_index": int(i),
                }
            )

        if not hits:
            return []

        hits.sort(key=lambda x: x["score"], reverse=True)

        if not dedupe_by_course:
            return hits[:top_k]

        deduped: list[dict[str, Any]] = []
        seen_courses: set[str] = set()

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
