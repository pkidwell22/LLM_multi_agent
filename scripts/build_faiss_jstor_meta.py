# scripts/build_faiss_jstor_meta.py
from __future__ import annotations
import argparse, gzip, json, re
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())

# Use your existing embed utils
from rag.embeddings import load_embedder, encode_texts

def safe(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def to_text(rec: Dict) -> str:
    title   = safe(rec.get("title") or rec.get("article_title") or rec.get("name") or "")
    abstr   = safe(rec.get("abstract") or rec.get("summary") or "")
    journal = safe(rec.get("journal") or rec.get("container_title") or rec.get("source") or "")
    year    = str(rec.get("year") or rec.get("pub_year") or rec.get("issued") or "")
    authors = rec.get("authors") or rec.get("author") or []
    if isinstance(authors, list):
        authors = ", ".join([safe(a.get("name") if isinstance(a, dict) else str(a)) for a in authors]) or ""
    else:
        authors = safe(str(authors))
    subjects = rec.get("subjects") or rec.get("keywords") or rec.get("topics") or []
    if isinstance(subjects, list):
        subjects = ", ".join(map(str, subjects))
    else:
        subjects = safe(str(subjects))

    lines = []
    if title:   lines.append(title)
    meta_line = " • ".join([p for p in [authors, journal, year] if p])
    if meta_line: lines.append(meta_line)
    if subjects: lines.append(f"Subjects: {subjects}")
    if abstr:    lines.append(f"Abstract: {abstr}")
    return "\n".join(lines).strip()

def pick_id(rec: Dict, fallback_idx: int) -> str:
    for k in ("stable_id","stableId","id","doi","identifier","handle"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # derive from stable_url if present
    for k in ("stable_url","stableUrl","url"):
        v = rec.get(k)
        if isinstance(v, str) and "jstor.org/stable" in v:
            return v.split("/stable/")[-1].split("?")[0].strip("/")
    return f"jstormeta:{fallback_idx}"

def load_jsonl(path: Path, limit: int | None = None) -> pd.DataFrame:
    open_fn = gzip.open if path.suffix == ".gz" else open
    rows = []
    with open_fn(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = pick_id(rec, i)
            txt = to_text(rec)
            if not txt:
                continue
            rows.append({
                "record_id": rid,
                "text": txt,
                "title": safe(rec.get("title") or rec.get("article_title") or ""),
                "journal": safe(rec.get("journal") or rec.get("container_title") or ""),
                "year": rec.get("year") or rec.get("pub_year") or "",
                "stable_url": rec.get("stable_url") or rec.get("stableUrl") or rec.get("url") or "",
                "doi": rec.get("doi") or "",
            })
            if limit and len(rows) >= limit:
                break
    return pd.DataFrame(rows)

def build_index(df: pd.DataFrame, out_dir: Path, train_cap: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    model_name = CFG["embeddings"]["model"]
    device     = str(CFG["embeddings"].get("device","auto"))
    batch_size = int(CFG["embeddings"].get("batch_size",64))

    texts = df["text"].tolist()
    ids   = df["record_id"].tolist()

    embedder = load_embedder(model_name, device)
    vecs = encode_texts(embedder, texts, batch_size=batch_size).astype(np.float32)

    # write meta
    pq.write_table(pa.Table.from_pandas(df.drop(columns=["text"])), out_dir / "meta.parquet")

    # optional: cache embeddings for reuse
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"record_id": ids, "vec": list(vecs)})),
                   cache / "embeddings.parquet")

    # build faiss
    spec = CFG["faiss"]["jstor_meta"]
    dim  = int(spec["dim"])
    index = faiss.index_factory(dim, str(spec["factory"]))

    # train if IVF/PQ
    if hasattr(index, "is_trained") and not index.is_trained:
        train = vecs
        if train_cap and len(train) > train_cap:
            idx = np.random.default_rng(42).choice(len(train), size=train_cap, replace=False)
            train = train[idx]
        print(f"[info] training ({spec['factory']}) on {len(train):,} vectors…")
        index.train(train)

    print(f"[info] adding {len(vecs):,} vectors…")
    index.add(vecs)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    print(f"[ok] built jstor_meta → {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to JSTOR metadata (.jsonl or .jsonl.gz)")
    ap.add_argument("--limit", type=int, default=0, help="Only ingest first N rows (0 = all)")
    ap.add_argument("--train-cap", type=int, default=200_000, help="Vectors to sample for IVF training (0 = all)")
    args = ap.parse_args()

    src = Path(args.jsonl)
    df = load_jsonl(src, limit=(args.limit or None))
    if df.empty:
        raise SystemExit("No usable metadata rows found.")

    out_dir = Path(CFG["faiss"]["jstor_meta"]["path"])
    cap = None if args.train_cap == 0 else args.train_cap
    build_index(df, out_dir, train_cap=cap)

if __name__ == "__main__":
    main()



