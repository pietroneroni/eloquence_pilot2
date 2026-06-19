# Build_rag_index.py
import re
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Dataset / language config
# -----------------------------

EMBED_MODEL = "intfloat/multilingual-e5-base"
SUPPORTED_LANGS = ("eng", "ita")

# Default project layout:
#   Sdialog/
#     RAG_University_Eng/
#       merged_outputUniversità*_translated_en.txt
#       Embeddings/
#     RAG_University_Ita/
#       merged_outputUniversità*.txt
#       Embeddings/
DEFAULT_DATASET_DIRS = {
    "eng": "RAG_University_Eng",
    "ita": "RAG_University_Ita",
}

DEFAULT_INPUT_PATTERNS = {
    "eng": ["merged_outputUniversità*_translated_en.txt"],
    "ita": ["merged_outputUniversità*.txt"],
}

# -----------------------------
# Parsing config
# -----------------------------

HEADER_ALIASES = {
    "UNIVERSITÀ": "UNIVERSITY",
    "UNIVERSITY": "UNIVERSITY",

    "CORSO": "COURSE",
    "COURSE": "COURSE",

    "TIPO": "TYPE",
    "TYPE": "TYPE",

    "CURRICULUM": "CURRICULUM",

    "SEZIONE": "SECTION",
    "SECTION": "SECTION",

    "URL": "URL",
}

HEADER_RE = re.compile(
    rf"^({'|'.join(map(re.escape, HEADER_ALIASES.keys()))})\s*:\s*(.*)\s*$"
)

CANONICAL_HEADER_KEYS = ["UNIVERSITY", "COURSE", "TYPE", "CURRICULUM", "SECTION", "URL"]

# Map file-stem -> region. The normalize_stem() function removes suffixes like
# _translated_en, so the same mapping works for both Italian and English files.
FILESTEM_TO_REGION = {
    "merged_outputUniversitàAquila": "Abruzzo",
    "merged_outputUniversitàChieti": "Abruzzo",
    "merged_outputUniversitàFoggia": "Puglia",
    "merged_outputUniversitàMessina": "Sicilia",
    "merged_outputUniversitàSalerno": "Campania",
    "merged_outputUniversitàSassari": "Sardegna",
    "merged_outputUniversitàSiena": "Toscana",
    "merged_outputUniversitàTrento": "Trentino-Alto Adige",
}

REGION_TO_MACRO = {
    "Valle d'Aosta": "nord",
    "Piemonte": "nord",
    "Liguria": "nord",
    "Lombardia": "nord",
    "Trentino-Alto Adige": "nord",
    "Veneto": "nord",
    "Friuli-Venezia Giulia": "nord",
    "Emilia-Romagna": "nord",
    "Toscana": "centro",
    "Umbria": "centro",
    "Marche": "centro",
    "Lazio": "centro",
    "Abruzzo": "centro",
    "Molise": "sud",
    "Campania": "sud",
    "Puglia": "sud",
    "Basilicata": "sud",
    "Calabria": "sud",
    "Sicilia": "isole",
    "Sardegna": "isole",
}

# Drop obvious noise / test blocks
_TEST_NOISE_RX = re.compile(r"\b(testare il rag|nome corso di test|example\.test)\b", re.I)

# -----------------------------
# Data model
# -----------------------------

@dataclass
class Chunk:
    id: str
    text: str
    meta: dict


# -----------------------------
# I/O and dataset helpers
# -----------------------------

def normalize_language(lang: str | None) -> str:
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


def default_project_root() -> Path:
    # Project root: Sdialog/
    return Path(__file__).resolve().parents[1]


def default_rag_dir(project_root: Path, lang: str) -> Path:
    lang = normalize_language(lang)
    preferred = project_root / DEFAULT_DATASET_DIRS[lang]

    # Backward compatibility with the old layout. Useful until you rename the
    # current RAG_University folder to RAG_University_Eng.
    if lang == "eng" and not preferred.exists():
        legacy = project_root / "RAG_University"
        if legacy.exists():
            return legacy

    return preferred


def discover_input_paths(
    rag_dir: str | Path,
    *,
    lang: str = "eng",
    input_glob: str | None = None,
) -> list[Path]:
    rag_dir = Path(rag_dir)
    lang = normalize_language(lang)

    patterns = [input_glob] if input_glob else DEFAULT_INPUT_PATTERNS[lang]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(rag_dir.glob(pattern))

    # In the Italian folder, keep the original Italian files and avoid indexing
    # English translated copies if they are accidentally present.
    if lang == "ita" and not input_glob:
        paths = [
            p for p in paths
            if not p.stem.endswith("_translated_en") and not p.stem.endswith("_en")
        ]

    return sorted(dict.fromkeys(paths))


def read_text(path: str) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="ignore")


# -----------------------------
# Parsing helpers
# -----------------------------

def iter_blocks(text: str) -> Iterable[str]:
    # split on lines like "-----"
    parts = re.split(r"\n\s*-{5,}\s*\n", text)
    for p in parts:
        p = p.strip()
        if p:
            yield p


def parse_block(block: str):
    lines = [l.rstrip() for l in block.splitlines()]
    meta, content_lines = {}, []
    in_header = True

    for line in lines:
        m = HEADER_RE.match(line)
        if in_header and m:
            raw_key = m.group(1)
            key = HEADER_ALIASES[raw_key]
            meta[key] = m.group(2).strip()
        else:
            in_header = False
            content_lines.append(line)

    content = "\n".join(content_lines).strip()

    if "COURSE" in meta:
        m = re.match(r"^\[(?P<code>[^\]]+)\]\s*(?P<name>.*)$", meta["COURSE"])
        if m:
            meta["COURSE_CODE"] = m.group("code").strip()
            meta["COURSE_NAME"] = m.group("name").strip()

    return meta, content


def should_keep_block(meta: dict, content: str) -> bool:
    if not meta.get("UNIVERSITY") or not meta.get("COURSE"):
        return False

    if _TEST_NOISE_RX.search(content or ""):
        return False

    if len((content or "").strip()) < 50:
        return False

    return True


def normalize_stem(stem: str) -> str:
    for suffix in ("_translated_en", "_en", "_translated"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def chunk_text(text: str, max_chars=1200, overlap=150):
    """
    max_chars: tienilo piu basso se noti che molta roba viene troncata dal modello.
    E5 gestisce bene ma resta un encoder con max length.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for para in paras:
        cand = para if not buf else buf + "\n\n" + para
        if len(cand) <= max_chars:
            buf = cand
            continue

        if buf:
            chunks.append(buf)

        if len(para) > max_chars:
            sentences = re.split(r"(?<=[\.\?\!])\s+", para)
            tmp = ""
            for s in sentences:
                cand2 = s if not tmp else tmp + " " + s
                if len(cand2) <= max_chars:
                    tmp = cand2
                else:
                    if tmp:
                        chunks.append(tmp)
                    tmp = s
            if tmp:
                chunks.append(tmp)
            buf = ""
        else:
            buf = para

    if buf:
        chunks.append(buf)

    if overlap > 0 and len(chunks) > 1:
        out, prev = [], ""
        for c in chunks:
            if prev:
                out.append(prev[-overlap:] + "\n" + c)
            else:
                out.append(c)
            prev = c
        return out

    return chunks


def batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def normalize_university_name(name: str) -> str:
    n = name.strip()
    n_upper = n.upper()

    if "AQUILA" in n_upper:
        return "University of L'Aquila"

    if "MESSINA" in n_upper:
        return "University of Messina"

    if "SALERNO" in n_upper:
        return "University of Salerno"

    if "SASSARI" in n_upper:
        return "University of Sassari"

    if "SIENA" in n_upper:
        return "University of Siena"

    if "TRENTO" in n_upper:
        return "University of Trento"

    if "FOGGIA" in n_upper:
        return "University of Foggia"

    if "D'ANNUNZIO" in n_upper or "ANNUNZIO" in n_upper or "CHIETI" in n_upper or "PESCARA" in n_upper:
        return "Gabriele d'Annunzio University of Chieti-Pescara"

    return n


def section_group(section: str) -> str:
    s = section.lower()

    if "general description" in s or "descrizione generale" in s:
        return "general_description"

    if "study plan" in s or "course plan" in s or "course structure" in s or "piani di studio" in s:
        return "study_plan"

    if "admission" in s or "requisiti" in s or "ammissione" in s:
        return "admission"

    if "objective" in s or "obiettivi" in s:
        return "learning_objectives"

    if "professional" in s or "sbocchi" in s:
        return "career_outcomes"

    return "other"


# -----------------------------
# Index build
# -----------------------------

def build_faiss_index(
    *,
    rag_dir: str | Path,
    embeddings_dir: str | Path | None = None,
    lang: str = "eng",
    input_glob: str | None = None,
    index_filename: str = "rag.index.faiss",
    chunks_filename: str = "rag.chunks.jsonl",
    embed_model: str = EMBED_MODEL,
) -> dict:
    lang = normalize_language(lang)
    rag_dir = Path(rag_dir)
    embeddings_dir = Path(embeddings_dir) if embeddings_dir else rag_dir / "Embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_input_paths(rag_dir, lang=lang, input_glob=input_glob)
    if not paths:
        pattern_msg = input_glob or ", ".join(DEFAULT_INPUT_PATTERNS[lang])
        raise SystemExit(f"Nessun file trovato in {rag_dir} con pattern: {pattern_msg}")

    out_index = embeddings_dir / index_filename
    out_chunks = embeddings_dir / chunks_filename

    print("RAG dir =", rag_dir)
    print("Embeddings dir =", embeddings_dir)
    print("Language =", lang)
    print("Matches =", [str(p) for p in paths])

    all_chunks: list[Chunk] = []

    for path in paths:
        text = read_text(str(path))
        stem = Path(path).stem
        base_stem = normalize_stem(stem)
        region = FILESTEM_TO_REGION.get(base_stem)

        for bi, block in enumerate(iter_blocks(text)):
            meta, content = parse_block(block)
            meta["LANGUAGE"] = lang

            if "UNIVERSITY" in meta:
                meta["UNIVERSITY_RAW"] = meta["UNIVERSITY"]
                meta["UNIVERSITY"] = normalize_university_name(meta["UNIVERSITY"])

            if not should_keep_block(meta, content):
                continue

            if region:
                meta["REGION"] = region
                meta["MACROAREA"] = REGION_TO_MACRO.get(region, "")

            if "SECTION" in meta:
                meta["SECTION_GROUP"] = section_group(meta["SECTION"])

            header_lines = []
            for k in CANONICAL_HEADER_KEYS:
                if k in meta:
                    header_lines.append(f"{k}: {meta[k]}")

            if "COURSE_CODE" in meta:
                header_lines.append(f"COURSE_CODE: {meta['COURSE_CODE']}")
            if "COURSE_NAME" in meta:
                header_lines.append(f"COURSE_NAME: {meta['COURSE_NAME']}")
            if "REGION" in meta:
                header_lines.append(f"REGION: {meta['REGION']}")
            if "MACROAREA" in meta:
                header_lines.append(f"MACROAREA: {meta['MACROAREA']}")
            header_lines.append(f"LANGUAGE: {lang}")

            header = "\n".join(header_lines).strip()
            content_chunks = chunk_text(content, max_chars=1200, overlap=120)

            for ci, ch in enumerate(content_chunks):
                chunk_id = f"{lang}__{stem}__b{bi:05d}__c{ci:03d}"
                chunk_text_final = (header + "\n\n" + ch).strip() if header else ch.strip()

                all_chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text_final,
                        meta={
                            **meta,
                            "source_file": Path(path).name,
                            "block_index": bi,
                            "chunk_index": ci,
                        },
                    )
                )

    if not all_chunks:
        raise SystemExit("Nessun chunk creato: controlla filtri/sezioni.")

    print("Files:", len(paths))
    print("Chunks:", len(all_chunks))

    from collections import Counter
    print("Sections:", Counter(c.meta.get("SECTION", "") for c in all_chunks).most_common(30))
    print("Universities:", Counter(c.meta.get("UNIVERSITY", "") for c in all_chunks).most_common(20))

    # ---- Embeddings (E5)
    print(f"Embedding model = {embed_model}")
    model = SentenceTransformer(embed_model)

    # IMPORTANT: for E5 use "passage:" for docs
    texts = [f"passage: {c.text}" for c in all_chunks]

    # Encode in batches to avoid RAM spikes
    vecs = []
    for batch in batched(texts, batch_size=128):
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vecs.append(np.asarray(embs, dtype="float32"))
    embeddings = np.vstack(vecs)

    # FAISS cosine via inner product on normalized vectors
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_index))

    with open(out_chunks, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps({"id": c.id, "text": c.text, "meta": c.meta}, ensure_ascii=False) + "\n")

    missing_course_code = sum(1 for c in all_chunks if not c.meta.get("COURSE_CODE"))
    print("Chunks without COURSE_CODE:", missing_course_code)
    print(f"OK: {len(all_chunks)} chunk indicizzati")
    print(f"Creati: {out_index} + {out_chunks}")

    return {
        "lang": lang,
        "rag_dir": str(rag_dir),
        "embeddings_dir": str(embeddings_dir),
        "input_files": [str(p) for p in paths],
        "chunks": len(all_chunks),
        "index_path": str(out_index),
        "chunks_path": str(out_chunks),
    }


def main():
    project_root = default_project_root()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        choices=SUPPORTED_LANGS,
        default="eng",
        help="Dataset language: eng uses RAG_University_Eng, ita uses RAG_University_Ita.",
    )
    parser.add_argument(
        "--rag-dir",
        default=None,
        help="Override dataset directory. If omitted, it is inferred from --lang.",
    )
    parser.add_argument(
        "--input-glob",
        default=None,
        help="Override input glob inside rag-dir, e.g. 'merged_outputUniversità*.txt'.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=None,
        help="Override embeddings output directory. Default: <rag-dir>/Embeddings.",
    )
    parser.add_argument(
        "--index-filename",
        default="rag.index.faiss",
    )
    parser.add_argument(
        "--chunks-filename",
        default="rag.chunks.jsonl",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
    )

    args = parser.parse_args()
    lang = normalize_language(args.lang)
    rag_dir = Path(args.rag_dir) if args.rag_dir else default_rag_dir(project_root, lang)

    print("PROJECT_ROOT =", project_root)

    build_faiss_index(
        rag_dir=rag_dir,
        embeddings_dir=args.embeddings_dir,
        lang=lang,
        input_glob=args.input_glob,
        index_filename=args.index_filename,
        chunks_filename=args.chunks_filename,
        embed_model=args.embed_model,
    )


if __name__ == "__main__":
    main()
