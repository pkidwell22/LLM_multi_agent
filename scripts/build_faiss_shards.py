# scripts/build_faiss_shards.py
from __future__ import annotations
import argparse, math
from pathlib import Path
from typing import Tuple, Optional, Dict

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())

DATA = ROOT / "rag_store"
CHUNKS_PARQUET = DATA / "chunks" / "chunks.parquet"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _supports_training(index: faiss.Index) -> bool:
    return hasattr(index, "is_trained")


def _train_and_add(factory: str, dim: int, vecs: np.ndarray,
                   train_cap: Optional[int] = 200_000) -> faiss.Index:
    index = faiss.index_factory(dim, factory)
    if _supports_training(index) and not index.is_trained:
        if train_cap is not None and train_cap > 0 and len(vecs) > train_cap:
            idx = np.random.default_rng(42).choice(len(vecs), size=train_cap, replace=False)
            train_vecs = vecs[idx]
        else:
            train_vecs = vecs
        print(f"[info] training index ({factory}) on {len(train_vecs):,} vectors…")
        index.train(train_vecs)
    print(f"[info] adding {len(vecs):,} vectors to index…")
    index.add(vecs)
    return index


def _write_meta(out_dir: Path, chunk_ids: pd.Series) -> None:
    _ensure_dir(out_dir)
    pq.write_table(
        pa.Table.from_pandas(chunk_ids.reset_index(drop=True).to_frame(name="chunk_id")),
        out_dir / "meta.parquet",
    )


def _embed_resumable(df: pd.DataFrame, out_dir: Path,
                     model_name: str, device: str, batch_size: int,
                     reembed: bool = False) -> Path:
    """
    Embeds df['text'] → rows [chunk_id, vec(list<float32>)] into out_dir/embeddings.parquet.
    Resumable unless --reembed is passed.

    IMPORTANT: Torch is imported LAZILY here *only if* we actually need to embed,
    avoiding OpenMP runtime clashes when we do index-only rebuilds.
    """
    _ensure_dir(out_dir)
    emb_path = out_dir / "embeddings.parquet"

    done = 0
    if emb_path.exists() and not reembed:
        existing = pq.read_table(emb_path)
        done = existing.num_rows
        if done > 0:
            print(f"[resume] found {done} existing embeddings → {emb_path}")

    total = len(df)
    if done >= total and not reembed:
        print("[resume] embeddings already complete.")
        return emb_path

    # Lazy import only when we truly need to embed (prevents OpenMP clash)
    from rag.embeddings import load_embedder, encode_texts  # noqa: WPS433

    writer = pq.ParquetWriter(
        emb_path,
        pa.schema([("chunk_id", pa.string()), ("vec", pa.list_(pa.float32()))]),
        use_dictionary=False,
    )
    try:
        embedder = load_embedder(model_name, device)
        texts = df["text"].tolist()
        ids = df["chunk_id"].tolist()

        start = 0 if reembed else done
        if start >= total:
            return emb_path

        n_batches = math.ceil((total - start) / batch_size)
        pbar = tqdm(range(n_batches), total=n_batches, desc="Embedding")
        for step in pbar:
            a = start + step * batch_size
            b = min(a + batch_size, total)
            vecs = encode_texts(embedder, texts[a:b], batch_size=batch_size).astype(np.float32)
            batch_df = pd.DataFrame({"chunk_id": ids[a:b], "vec": list(vecs)})
            writer.write_table(pa.Table.from_pandas(batch_df))
    finally:
        writer.close()

    print(f"[ok] wrote embeddings → {emb_path}")
    return emb_path


def _load_embedded(emb_path: Path) -> Tuple[np.ndarray, pd.Series]:
    tbl = pq.read_table(emb_path)
    pdf = tbl.to_pandas()
    vecs = np.vstack(pdf["vec"].to_numpy()).astype(np.float32)
    chunk_ids = pdf["chunk_id"].reset_index(drop=True)
    return vecs, chunk_ids


def build_for_source(df_all: pd.DataFrame, source_key: str, source_label: str,
                     reembed: bool, rebuild_index: bool, train_cap: Optional[int]) -> None:
    sub = df_all[df_all["source"] == source_label].copy()
    if sub.empty:
        print(f"[warn] no data for '{source_key}'")
        return

    if "faiss" not in CFG or source_key not in CFG["faiss"]:
        raise KeyError(f"Missing FAISS config for shard '{source_key}' in settings.yaml")

    spec: Dict = CFG["faiss"][source_key]
    out_dir = Path(spec["path"])
    emb_dir = out_dir / "cache"

    model_name = CFG["embeddings"]["model"]
    device = str(CFG["embeddings"].get("device", "auto"))
    batch_size = int(CFG["embeddings"].get("batch_size", 64))

    _ensure_dir(out_dir)
    _ensure_dir(emb_dir)

    # This will NO-OP and avoid importing Torch if embeddings already exist and reembed=False
    print(f"[info] preparing embeddings for '{source_key}' (batch={batch_size}, device={device})…")
    emb_path = _embed_resumable(sub[["chunk_id", "text"]], emb_dir, model_name, device, batch_size,
                                reembed=reembed)

    print("[info] loading vectors from cache…")
    vecs, chunk_ids = _load_embedded(emb_path)

    idx_path = out_dir / "index.faiss"
    if idx_path.exists() and not rebuild_index:
        print(f"[resume] index already exists → {idx_path} (use --rebuild-index to rebuild)")
        _write_meta(out_dir, chunk_ids)
        return

    print(f"[info] building FAISS index at: {out_dir}")
    def train_and_add(factory, dim, v):
        return _train_and_add(factory, dim, v, train_cap=train_cap)

    index = train_and_add(str(spec["factory"]), int(spec["dim"]), vecs)
    faiss.write_index(index, str(idx_path))
    _write_meta(out_dir, chunk_ids)
    print(f"[ok] built {source_key} index → {idx_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="gutenberg",
                    help="Comma-separated shards to build (default: gutenberg)")
    ap.add_argument("--reembed", action="store_true",
                    help="Re-embed even if cache exists")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Rebuild FAISS even if index exists")
    ap.add_argument("--train-cap", type=int, default=200_000,
                    help="Max vectors to sample for IVF training (0 = use all)")
    args = ap.parse_args()

    if not CHUNKS_PARQUET.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PARQUET}")

    df_all = pd.read_parquet(CHUNKS_PARQUET)

    wanted = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    mapping = {
        "gutenberg": "gutenberg",
        "pdfs": "book",      # optional later
        "books": "book",     # legacy
        "jstor": "jstor",
        "reading": "reading",
    }

    cap = None if args.train_cap == 0 else args.train_cap

    for key, label in mapping.items():
        if key in wanted:
            build_for_source(
                df_all, key, label,
                reembed=args.reembed,
                rebuild_index=args.rebuild_index,
                train_cap=cap,
            )

    print("[ok] all requested indexes built/resumed.")


if __name__ == "__main__":
    main()
