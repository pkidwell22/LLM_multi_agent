# scripts/ingest_s2orc.py
"""
Ingest OA academic papers (default: allenai/peS2o v1) into parsed_docs.parquet.

Modes:
  --mode hf     : stream from Hugging Face (default dataset=allenai/peS2o, config=v1)
  --mode local  : read local JSONL(.gz) under rag_store/raw/s2orc/

Output:
  rag_store/parsed/parsed_docs.parquet rows:
    { doc_id, source, path, text, title, year, authors, venue }

Examples (PowerShell):
  # Default: peS2o v1
  python scripts/ingest_s2orc.py --mode hf --max_docs 5000

  # Explicit (other corpora)
  python scripts/ingest_s2orc.py --mode hf --dataset WINGNUS/ACL-OCL --split train --max_docs 20000
  python scripts/ingest_s2orc.py --mode hf --dataset pmc/open_access --max_docs 5000

  # Local S2ORC-style JSONL(.gz)
  python scripts/ingest_s2orc.py --mode local --max_docs 20000
"""
from __future__ import annotations
import argparse
import gzip
from pathlib import Path
from typing import Dict, Iterable, Optional

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "rag_store"
RAW_S2ORC = DATA / "raw" / "s2orc"
PARSED = DATA / "parsed"
PARSED.mkdir(parents=True, exist_ok=True)
OUT_PATH = PARSED / "parsed_docs.parquet"

def _normalize_authors(auth) -> str:
    if isinstance(auth, list):
        names = []
        for a in auth:
            if isinstance(a, dict) and "name" in a:
                names.append(str(a["name"]))
            else:
                names.append(str(a))
        return ", ".join([n for n in names if n])
    return str(auth) if auth else ""

def _extract_text_from_record(rec: Dict) -> str:
    """
    Prefer explicit full-text fields; fall back to body_text-like lists; then title+abstract.
    """
    # 1) Most common: 'text' (peS2o, PMC OA) or 'full_text' (ACL-OCL)
    for k in ["text", "full_text"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v

    # 2) Body-paragraph variants (S2ORC-style)
    parts = []
    for k in ["pdf_parse", "pdf_json", "grobid_parse", "body_text", "content"]:
        v = rec.get(k)
        if isinstance(v, list):
            body = "\n\n".join([str(x.get("text", "")) for x in v if isinstance(x, dict)])
            if body.strip():
                parts.append(body)
        elif isinstance(v, dict) and "body_text" in v:
            bt = v["body_text"]
            if isinstance(bt, list):
                body = "\n\n".join([str(x.get("text", "")) for x in bt if isinstance(x, dict)])
                if body.strip():
                    parts.append(body)
        elif isinstance(v, str) and v.strip():
            parts.append(v)

    if parts:
        return "\n\n".join(parts)

    # 3) Fallback: title + abstract
    title = rec.get("title") or rec.get("paper_title") or ""
    abstract = rec.get("abstract") or rec.get("paper_abstract") or ""
    return " ".join([title.strip(), abstract.strip()]).strip()

def _row_from_record(rec: Dict, source: str) -> Optional[Dict]:
    doc_id = (
        rec.get("paper_id")
        or rec.get("id")
        or rec.get("sha")
        or rec.get("doi")
        or rec.get("arxiv_id")
        or rec.get("s2_id")
        or rec.get("pmid")
        or rec.get("accession_id")
    )
    if not doc_id:
        return None

    text = _extract_text_from_record(rec)
    if not (text and text.strip()):
        return None

    row = {
        "doc_id": str(doc_id),
        "source": source,
        "path": "",  # not a local file
        "text": text,
        "title": rec.get("title") or rec.get("paper_title") or "",
        "year": rec.get("year") or rec.get("publication_year") or None,
        "authors": _normalize_authors(rec.get("authors")),
        "venue": rec.get("venue") or rec.get("journal") or rec.get("source") or "",
    }
    return row

def _append_rows(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    if OUT_PATH.exists():
        existing = pq.read_table(OUT_PATH)
        concat = pa.concat_tables([existing, table])
        pq.write_table(concat, OUT_PATH)
    else:
        pq.write_table(table, OUT_PATH)

# ---------------------- MODES ----------------------

def ingest_local(max_docs: int = 10000) -> int:
    RAW_S2ORC.mkdir(parents=True, exist_ok=True)
    files = sorted(list(RAW_S2ORC.glob("*.jsonl"))) + sorted(list(RAW_S2ORC.glob("*.jsonl.gz")))
    if not files:
        print(f"[warn] No local files in {RAW_S2ORC}. Place .jsonl or .jsonl.gz and retry.")
        return 0

    count = 0
    batch = []
    BATCH_N = 1000

    for fp in files:
        opener = gzip.open if (fp.suffix == ".gz" or fp.suffixes[-2:] == [".jsonl", ".gz"]) else open
        print(f"[info] reading {fp.name} ...")
        with opener(fp, "rb") as f:
            for raw in f:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "ignore")
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = orjson.loads(raw)
                except Exception:
                    continue
                row = _row_from_record(rec, source="s2orc_local")
                if row:
                    batch.append(row)
                    count += 1
                if len(batch) >= BATCH_N:
                    _append_rows(batch); batch.clear()
                if 0 < max_docs <= count:
                    break
        if batch:
            _append_rows(batch); batch.clear()
        if 0 < max_docs <= count:
            break

    print(f"[ok] ingested {count} records → {OUT_PATH}")
    return count

def ingest_hf(dataset: str, config: Optional[str], split: str, max_docs: int) -> int:
    try:
        from datasets import load_dataset
    except Exception:
        print("[error] 'datasets' not importable. Ensure versions: datasets==2.19.0, pyarrow==14.0.2, numpy==1.26.4, pandas==2.2.2")
        return 0

    print(f"[info] streaming from HF: {dataset} (config={config}, split={split}) ...")
    ds = load_dataset(dataset, config, split=split, streaming=True)

    # Source label for downstream visibility
    source_label = "pes2o" if dataset.lower().startswith("allenai/pes2o") else dataset

    count = 0
    batch = []
    BATCH_N = 1000

    for rec in ds:
        row = _row_from_record(rec, source=source_label)
        if row:
            batch.append(row)
            count += 1
        if len(batch) >= BATCH_N:
            _append_rows(batch); batch.clear()
        if 0 < max_docs <= count:
            break

    if batch:
        _append_rows(batch); batch.clear()

    print(f"[ok] ingested {count} records → {OUT_PATH}")
    return count

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "local"], required=True)
    ap.add_argument("--dataset", default="allenai/peS2o", help="HF dataset (default: allenai/peS2o)")
    ap.add_argument("--config", default="v1", help="HF config (peS2o: v1 is safest)")
    ap.add_argument("--split", default="train", help="HF split (most corpora use 'train')")
    ap.add_argument("--max_docs", type=int, default=10000, help="0=unlimited")
    args = ap.parse_args()

    if args.mode == "hf":
        ingest_hf(dataset=args.dataset, config=args.config, split=args.split, max_docs=args.max_docs)
    else:
        ingest_local(max_docs=args.max_docs)

if __name__ == "__main__":
    main()
