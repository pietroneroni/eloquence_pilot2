# Build_lancedb_index_university.py

import json
from pathlib import Path

import lancedb
import pandas as pd
import pyarrow as pa
import numpy as np
from sentence_transformers import SentenceTransformer

from Build_rag_index import (
    read_text,
    iter_blocks,
    parse_block,
    should_keep_block,
    normalize_stem,
    normalize_university_name,
    section_group,
    chunk_text,
    FILESTEM_TO_REGION,
    REGION_TO_MACRO,
    CANONICAL_HEADER_KEYS,
    EMBED_MODEL,
    SUPPORTED_LANGS,
    DEFAULT_INPUT_PATTERNS,
    default_project_root,
    default_rag_dir,
    discover_input_paths,
    normalize_language,
)

VECTOR_DIM = 768


def build_lancedb_index(
    rag_dir: str,
    lancedb_dir: str,
    table_name: str = "universities",
    *,
    lang: str = "eng",
    input_glob: str | None = None,
    embed_model: str = EMBED_MODEL,
):
    lang = normalize_language(lang)
    rag_dir = Path(rag_dir)
    db = lancedb.connect(lancedb_dir)

    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("id", pa.string()),
        pa.field("university", pa.string()),
        pa.field("course", pa.string()),
        pa.field("course_code", pa.string()),
        pa.field("section", pa.string()),
        pa.field("region", pa.string()),
        pa.field("macroarea", pa.string()),
        pa.field("language", pa.string()),
    ])

    tbl = db.create_table(table_name, schema=schema, mode="overwrite")

    model = SentenceTransformer(embed_model)
    rows = []

    paths = discover_input_paths(rag_dir, lang=lang, input_glob=input_glob)
    if not paths:
        pattern_msg = input_glob or ", ".join(DEFAULT_INPUT_PATTERNS[lang])
        raise SystemExit(f"No RAG input files found in {rag_dir} with pattern: {pattern_msg}")

    print("RAG dir =", rag_dir)
    print("LanceDB dir =", lancedb_dir)
    print("Table =", table_name)
    print("Language =", lang)
    print("Matches =", [str(p) for p in paths])

    for path in paths:
        text = read_text(str(path))
        stem = path.stem
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

            for ci, ch in enumerate(chunk_text(content, max_chars=1200, overlap=120)):
                chunk_id = f"{lang}__{stem}__b{bi:05d}__c{ci:03d}"
                final_text = (header + "\n\n" + ch).strip()

                rows.append({
                    "id": chunk_id,
                    "text": final_text,
                    "metadata": json.dumps({
                        **meta,
                        "source_file": path.name,
                        "block_index": bi,
                        "chunk_index": ci,
                    }, ensure_ascii=False),
                    "university": meta.get("UNIVERSITY", ""),
                    "course": meta.get("COURSE", ""),
                    "course_code": meta.get("COURSE_CODE", ""),
                    "section": meta.get("SECTION", ""),
                    "region": meta.get("REGION", ""),
                    "macroarea": meta.get("MACROAREA", ""),
                    "language": lang,
                })

    if not rows:
        raise SystemExit("No rows created: check filters/sections.")

    batch_size = 128

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [f"passage: {r['text']}" for r in batch]
        vecs = model.encode(texts, normalize_embeddings=True)

        for r, v in zip(batch, vecs):
            r["vector"] = np.asarray(v, dtype="float32").tolist()

        tbl.add(pd.DataFrame(batch))

    print(f"OK: created LanceDB table '{table_name}' in {lancedb_dir}")
    print(f"Rows: {len(rows)}")
    return table_name


if __name__ == "__main__":
    import argparse

    project_root = default_project_root()

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
        help="Override input glob inside rag-dir.",
    )
    parser.add_argument(
        "--lancedb-dir",
        default=None,
        help="Override LanceDB output directory. Default: <rag-dir>/Embeddings/lancedb.",
    )
    parser.add_argument(
        "--table-name",
        default="universities",
    )
    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL,
    )

    args = parser.parse_args()
    lang = normalize_language(args.lang)
    rag_dir = Path(args.rag_dir) if args.rag_dir else default_rag_dir(project_root, lang)
    lancedb_dir = Path(args.lancedb_dir) if args.lancedb_dir else rag_dir / "Embeddings" / "lancedb"
    lancedb_dir.mkdir(parents=True, exist_ok=True)

    table_name = build_lancedb_index(
        rag_dir=str(rag_dir),
        lancedb_dir=str(lancedb_dir),
        table_name=args.table_name,
        lang=lang,
        input_glob=args.input_glob,
        embed_model=args.embed_model,
    )

    print(f"OK: created LanceDB table '{table_name}' in {lancedb_dir}")
