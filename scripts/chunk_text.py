# scripts/chunk_text.py
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "settings.yaml"

def load_cfg():
    if CFG_PATH.exists():
        cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
    else:
        cfg = {}
    # sensible defaults if keys are missing
    cfg.setdefault("embeddings", {}).setdefault("model", "BAAI/bge-base-en-v1.5")
    return cfg

CFG = load_cfg()

DATA = ROOT / "rag_store"
PARSED = DATA / "parsed" / "parsed_docs.parquet"
CHUNKS = DATA / "chunks"
CHUNKS.mkdir(parents=True, exist_ok=True)

# Use the embedding model's tokenizer for stable token windows
TOKENIZER = AutoTokenizer.from_pretrained(CFG["embeddings"]["model"])

def _chunk_tokens(tokens, max_len: int, overlap: int):
    step = max_len - overlap
    i = 0
    n = max(1, len(tokens))
    while i < n:
        j = min(i + max_len, n)
        yield tokens[i:j], i, j
        if j == n:
            break
        i += step

def main():
    if not PARSED.exists():
        raise FileNotFoundError(f"Missing parsed docs file: {PARSED}")
    df = pd.read_parquet(PARSED)
    if df.empty:
        print("[warn] parsed_docs.parquet is empty; nothing to chunk.")
        return

    out_rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="chunking"):
        text = r.get("text") or ""
        if not text.strip():
            continue
        toks = TOKENIZER.encode(text, add_special_tokens=False)

        # Source-aware windowing; default to 'article' sizing if missing
        # Source-aware windowing; treat Gutenberg like books
        src = (r.get("source") or "").lower()
        book_like = {"book", "gutenberg"}

        # Optional: allow overrides from settings.yaml
        chunk_cfg = (CFG.get("chunking") or {})
        book_cfg = (chunk_cfg.get("book") or {})
        art_cfg = (chunk_cfg.get("article") or {})

        if src in book_like:
            max_len = int(book_cfg.get("max_len", 1100))
            overlap = int(book_cfg.get("overlap", 200))
        else:
            max_len = int(art_cfg.get("max_len", 800))
            overlap = int(art_cfg.get("overlap", 150))

        for toks_slice, a, b in _chunk_tokens(toks, max_len, overlap):
            chunk_text = TOKENIZER.decode(toks_slice, skip_special_tokens=True)
            out_rows.append(
                {
                    "chunk_id": f'{r.get("doc_id", "doc")}:{a}-{b}',
                    "doc_id": r.get("doc_id", "doc"),
                    "source": r.get("source", "article"),
                    "path": r.get("path", ""),
                    "text": chunk_text,
                    "tok_start": a,
                    "tok_end": b,
                }
            )

    out_path = CHUNKS / "chunks.parquet"
    if out_rows:
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(out_rows)), out_path)
        print(f"[ok] chunks: {len(out_rows)} → {out_path}")
    else:
        print("[warn] no chunks produced (no non-empty texts found).")

if __name__ == "__main__":
    main()
