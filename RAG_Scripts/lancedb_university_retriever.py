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

    return preferred


def _metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_meta = row.get("metadata") or row.get("meta") or "{}"
    if isinstance(raw_meta, dict):
        meta = dict(raw_meta)
    else:
        try:
            meta = json.loads(raw_meta)
        except Exception:
            return {}
    if not isinstance(meta, dict):
        return {}
    return meta


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
        self._course_evidence_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _course_evidence(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Read declarations on the same course, never infer its city from the university."""
        university = str(meta.get("UNIVERSITY") or "").strip()
        code = str(meta.get("COURSE_CODE") or "").strip()
        course = str(meta.get("COURSE") or "").strip()
        key = (_norm(university), _norm(code or course))
        if key in self._course_evidence_cache:
            return self._course_evidence_cache[key]
        column, value = ("course_code", code) if code else ("course", course)
        escaped = value.replace("'", "''")
        rows = self.table.search().where(f"{column} = '{escaped}'").select(["metadata", "text"]).limit(self.table.count_rows()).to_list()
        declarations: dict[str, str] = {}
        degree_classes: set[str] = set()
        cities: set[str] = set()
        sources: set[str] = set()
        for row in rows:
            other = _metadata_from_row(row)
            if _norm(str(other.get("UNIVERSITY") or "")) != _norm(university):
                continue
            text = str(row.get("text") or "")
            declared = next((str(other[k]).strip() for k in ("DEGREE_TYPE", "TIPO", "TYPE") if other.get(k)), "")
            if not declared:
                title = str(other.get("COURSE") or "").strip()
                title_rx = r"\s+".join(re.escape(word) for word in title.split())
                match = re.search(
                    rf"(?im)^\s*{title_rx}[ \t]+"
                    r"((?:Corso di )?Laurea(?: Magistrale(?: a ciclo unico)?| Triennale)?"
                    r"|Bachelor(?:'s)? degree|Master(?:'s)? degree|Dottorato|PhD)\b", text,
                ) if title else None
                if not match and code:
                    # Catalogue titles can include curriculum/language suffixes absent from the heading.
                    match = re.search(
                        rf"(?im)^\s*\[{re.escape(code)}\][^\n]*?\s+"
                        r"((?:Corso di )?Laurea(?: Magistrale(?: a ciclo unico)?| Triennale)?"
                        r"|Corso di Dottorato|Corso di Perfezionamento|Corso di Alta Formazione"
                        r"|Master universitario di (?:Primo|Secondo) Livello)\b", text,
                    )
                if not match:
                    match = re.search(
                        r"(?im)^\s*((?:Corso di )?Laurea(?: Magistrale(?: a ciclo unico)?| Triennale)?"
                        r"|Corso di Dottorato|Master universitario di (?:Primo|Secondo) Livello"
                        r"|Corso di Perfezionamento|Corso di Alta Formazione)\b",
                        text,
                    )
                declared = match.group(1) if match else ""
            for match in re.finditer(
                r"(?im)^\s*Classe(?: di laurea)?\s*:?\s*(L(?:M|MG)?\s*[-/–‑]\s*(?:SNT\s*)?\d+)\b", text,
            ):
                degree_classes.add("Classe di laurea: " + match.group(1))
            if declared:
                declarations[re.sub(r"^corso di\s+", "", _norm(declared))] = declared
            for name in ("CITY", "CITTA", "CITTÀ", "CAMPUS_CITY", "CITIES"):
                if isinstance(other.get(name), str) and other[name].strip():
                    cities.add(other[name].strip())
                elif isinstance(other.get(name), list):
                    cities.update(str(value).strip() for value in other[name] if str(value).strip())
            for match in re.finditer(r"(?im)^\s*(?:sede didattica|sede del corso|teaching location|campus city)\s*:\s*([^\n]+)", text):
                cities.add(match.group(1).strip())
            for match in re.finditer(r"(?im)^\s*Sede\s*:\s*([^\n]+)", text):
                cities.add(match.group(1).strip())
            # Structured inline catalogue field, not an incidental city or the university name.
            for match in re.finditer(r"\bSede\s+([^\n]+?)\s+Dipartimento\b", text):
                cities.add(match.group(1).strip())
            if other.get("URL"):
                sources.add(str(other["URL"]))
        if not declarations:
            declarations = {_norm(value): value for value in degree_classes}
        evidence = {
            "TYPE": next(iter(declarations.values())) if len(declarations) == 1 else "",
            "TYPE_STATUS": "declared" if len(declarations) == 1 else "conflicting" if declarations else "unknown",
            "CITIES": sorted(cities), "SOURCES": sorted(sources),
        }
        self._course_evidence_cache[key] = evidence
        return evidence

    def search(
        self, query: str, *, top_k: int = 12,
        region: str | None = None, macroarea: str | None = None,
        section_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic retrieval with prefilters, without geographic bonuses or title heuristics."""
        if not query.strip():
            return []
        vector = self.model.encode([f"query: {query.strip()}"], normalize_embeddings=True)
        vector = np.asarray(vector[0], dtype="float32")
        clauses = []
        for column, value in (("region", region), ("macroarea", macroarea)):
            if value:
                escaped = value.replace("'", "''")
                clauses.append(f"{column} = '{escaped}'")
        if section_contains:
            escaped = section_contains.replace("'", "''")
            clauses.append(f"section LIKE '%{escaped}%'")
        search = self.table.search(vector, vector_column_name="vector")
        if clauses:
            search = search.where(" AND ".join(clauses), prefilter=True)
        # Bound chunks read; the caller applies the single eligible-candidate cap.
        rows = search.limit(max(top_k * 30, 100)).to_list()
        courses: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            meta = _metadata_from_row(row)
            university = str(meta.get("UNIVERSITY") or "").strip()
            course = str(meta.get("COURSE") or "").strip()
            key = (_norm(university), _norm(str(meta.get("COURSE_CODE") or course)))
            if not university or not course:
                print('[RAG][reject] ' + json.dumps({"version":"courses_v3", "id":row.get("id"), "reason":"missing_course_identity"}, ensure_ascii=False))
                continue
            if key in courses:
                continue
            meta.update(self._course_evidence(meta))
            body = re.sub(r"(?m)^[A-Z_À]+:[^\n]*\n?", "", str(row.get("text") or "")).strip()
            courses[key] = {"key":key, "score":-float(row.get("_distance", 0)),
                            "meta":meta, "text":body, "id":row.get("id")}
        return list(courses.values())
