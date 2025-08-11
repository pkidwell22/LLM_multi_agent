# scripts/ingest_hf_academic.py
"""
Ingest HF datasets known to exist: scientific_papers:{arxiv,pubmed}, pg19.

Usage (PowerShell):
  # 5k arXiv papers (title+abstract+body)
  python scripts/ingest_hf_academic.py --dataset scientific_papers --config arxiv --max_docs 5000

  # 5k PubMed long-form articles
  python scripts/ingest_hf_academic.py --dataset scientific_papers --config pubmed --max_docs 5000

  # PG-19 books (good for 'Western canon' dry run)
  python scripts/ingest_hf_academic.py --dataset pg19 --max_docs 2000
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "rag_store"
PARSED = DATA / "parsed"
PARSED.mkdir(parents=True, exist_ok=True)
OUT = PARSED / "parsed_docs.parquet"

def _append(rows: List[Dict]):
    if not rows:
        return
    table = pa.Table.from_pandas(pd.DataFrame(rows))
    if OUT.exists():
        existing = pq.read_table(OUT)
        pq.write_table(pa.concat_tables([existing, table]), OUT)
    else:
        pq.write_table(table, OUT)

def _row_scientific_papers(rec: Dict, source_tag: str) -> Optional[Dict]:
    # scientific_papers has fields: 'article', 'abstract', 'title', 'paper_id'
    text = (rec.get("article") or "").strip()
    if not text:
        text = (" ".join([rec.get("title") or "", rec.get("abstract") or ""])).strip()
    if not text:
        return None
    return {
        "doc_id": str(rec.get("paper_id") or rec.get("title") or hash(text)),
        "source": source_tag,
        "path": "",
        "text": text,
        "title": rec.get("title") or "",
        "year": None,
        "authors": "",
        "venue": "",
    }

def _row_pg19(rec: Dict) -> Optional[Dict]:
    # pg19 fields: 'book_id', 'text'
    text = (rec.get("text") or "").strip()
    if not text:
        return None
    return {
        "doc_id": str(rec.get("book_id")),
        "source": "pg19",
        "path": "",
        "text": text,
        "title": str(rec.get("book_id")),
        "year": None,
        "authors": "",
        "venue": "",
    }

def ingest(dataset: str, config: Optional[str], split: str, max_docs: int):
    from datasets import load_dataset  # import here to surface env errors immediately

    print(f"[info] loading HF dataset: {dataset} (config={config}, split={split}) …")
    ds = load_dataset(dataset, config, split=split, streaming=True)

    rows: List[Dict] = []
    batch_n = 1000
    n = 0

    for rec in ds:
        if dataset == "scientific_papers":
            source_tag = f"scientific_papers_{config or 'unknown'}"
            row = _row_scientific_papers(rec, source_tag)
        elif dataset == "pg19":
            row = _row_pg19(rec)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        if row:
            rows.append(row)
            n += 1

        if len(rows) >= batch_n:
            _append(rows)
            rows.clear()
        if 0 < max_docs <= n:
            break

    if rows:
        _append(rows)

    print(f"[ok] ingested {n} items → {OUT}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["scientific_papers", "pg19"])
    ap.add_argument("--config", default=None, help="For scientific_papers: arxiv or pubmed")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_docs", type=int, default=5000)
    args = ap.parse_args()
    ingest(args.dataset, args.config, args.split, args.max_docs)

if __name__ == "__main__":
    main()
