# scripts/ingest_philosophy.py
from __future__ import annotations
import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = ROOT / "rag_store" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)
OUT = PARSED_DIR / "parsed_docs.parquet"


def _append_rows(rows: List[Dict]):
    """Append rows into OUT Parquet (create if missing)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    if OUT.exists():
        existing = pq.read_table(OUT)
        pq.write_table(pa.concat_tables([existing, table]), OUT)
    else:
        pq.write_table(table, OUT)


# -------------------
# A) Gutenberg (philosophy)
# -------------------
def ingest_gutenberg_philosophy(max_books=600, min_words=12000, sleep_s=0.5):
    url = "https://gutendex.com/books"
    params = {"topic": "philosophy"}
    seen, rows = 0, []

    while seen < max_books:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for book in data.get("results", []):
            if seen >= max_books:
                break

            txt_url = next(
                (u for u, mt in book["formats"].items() if "text/plain" in mt and "zip" not in u),
                None,
            )
            if not txt_url:
                continue

            try:
                txt = requests.get(txt_url, timeout=60).text
            except Exception:
                continue

            if len(txt.split()) < min_words:
                continue

            rows.append(
                {
                    "doc_id": f"gutenberg_{book['id']}",
                    "source": "gutenberg",
                    "path": txt_url,
                    "text": txt,
                    "title": book.get("title", ""),
                    "year": book.get("authors", [{}])[0].get("birth_year"),
                    "authors": ", ".join(a.get("name", "") for a in book.get("authors", [])),
                    "venue": "Project Gutenberg",
                }
            )
            seen += 1

        if not data.get("next"):
            break
        url = data["next"]
        time.sleep(sleep_s)

    _append_rows(rows)
    print(f"[ok] Gutenberg (philosophy): {len(rows)} docs appended → {OUT}")


# -------------------
# B) peS2o (OA papers)
# -------------------
PHILO_PAT = re.compile(
    r"\b("
    r"philosophy|philosophical|ethic|metaphys|epistemolog|logic|aestheti|"
    r"phenomenolog|hermeneutic|ontology|stoic|aristotle|plato|kant|hegel|"
    r"nietzsche|wittgenstein"
    r")\b",
    re.I,
)


def _record_to_row(rec: Dict) -> Optional[Dict]:
    doc_id = rec.get("paper_id") or rec.get("id") or rec.get("sha") or rec.get("doi")
    if not doc_id:
        return None

    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    # various dumps expose body differently
    body = rec.get("body") or rec.get("content") or rec.get("text") or ""
    text = body or f"{title}\n\n{abstract}"

    # quick topical filter so we don’t ingest the entire peS2o
    hay = (title + " " + abstract + " " + text[:5000])
    if not text or not PHILO_PAT.search(hay):
        return None

    authors = rec.get("authors", "")
    if isinstance(authors, list):
        authors = ", ".join(map(str, authors))

    return {
        "doc_id": f"pes2o_{doc_id}",
        "source": "pes2o",
        "path": "",  # streamed JSON; not a local file
        "text": text,
        "title": title,
        "year": rec.get("year"),
        "authors": authors,
        "venue": rec.get("venue") or rec.get("journal") or "peS2o",
    }


def _list_shards(repo_id: str) -> List[str]:
    """List JSON(.gz) shard paths under data/v2 or data/v1 on HF for a dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return []

    # prefer v2 layout; fallback to v1
    v2 = [f for f in files if f.startswith("data/v2/") and f.endswith(".json.gz")]
    if v2:
        return sorted(v2)
    v1 = [f for f in files if f.startswith("data/v1/") and f.endswith(".json.gz")]
    return sorted(v1)


def ingest_pes2o_hf(max_docs=20000):
    """Stream peS2o shards directly from the Hub (repo_type=dataset), with fallbacks."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_url

    print(f"[info] peS2o: streaming from Hub… (max_docs={max_docs})")

    # Try multiple repo ids to avoid casing/mirror issues.
    candidate_repos = [
        "allenai/peS2o",
        "allenai/pes2o",
        "allenai/peS2o-mini",
    ]

    urls: List[str] = []
    used_repo = None
    for rid in candidate_repos:
        shards = _list_shards(rid)
        if shards:
            used_repo = rid
            # build signed file URLs with the correct repo_type
            urls = [
                hf_hub_url(repo_id=rid, filename=fp, repo_type="dataset")
                for fp in shards
            ]
            break

    if not urls:
        # absolute fallback: try the dataset script directly (if present)
        for rid in candidate_repos:
            try:
                ds = load_dataset(rid, split="train", streaming=True)
                used_repo = rid
                print(f"[warn] falling back to dataset script: {rid}")
                break
            except Exception:
                ds = None
        if ds is None:
            raise FileNotFoundError(
                "Could not locate peS2o shards or dataset script on the Hub. "
                "Try again later or provide a local JSONL(.gz) under rag_store/raw/s2orc/."
            )
    else:
        # JSON loader on explicit file URLs
        ds = load_dataset("json", data_files=urls, split="train", streaming=True)

    print(f"[info] peS2o repo: {used_repo}")

    batch, kept = [], 0
    for rec in ds:
        row = _record_to_row(rec)
        if row:
            batch.append(row)
            kept += 1
        if len(batch) >= 1000:
            _append_rows(batch)
            batch.clear()
            print(f"[info] peS2o appended: {kept}")
        if kept >= max_docs:
            break

    if batch:
        _append_rows(batch)

    print(f"[ok] peS2o done: {kept} rows → {OUT}")


# -------------------
# CLI
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_gutenberg", action="store_true")
    ap.add_argument("--skip_pes2o", action="store_true")
    ap.add_argument("--pes2o_max", type=int, default=20000, help="Max peS2o docs to fetch")
    args = ap.parse_args()

    if not args.skip_gutenberg:
        ingest_gutenberg_philosophy()
    if not args.skip_pes2o:
        ingest_pes2o_hf(args.pes2o_max)

    print("[ok] Philosophy corpus ingestion complete.")


if __name__ == "__main__":
    main()
