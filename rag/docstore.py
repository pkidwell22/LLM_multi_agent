# rag/docstore.py
from pathlib import Path
import pandas as pd

class DocStore:
    """Loads chunk metadata and lets you look up chunk → {doc_id, source, path, text, tok_start, tok_end}."""

    def __init__(self, chunks_parquet: Path):
        if not Path(chunks_parquet).exists():
            raise FileNotFoundError(f"Missing chunks file: {chunks_parquet}")
        self.df = pd.read_parquet(chunks_parquet)
        self.df.set_index("chunk_id", inplace=True)

    def get_chunk(self, chunk_id: str) -> dict:
        row = self.df.loc[chunk_id]
        return {
            "chunk_id": chunk_id,
            "doc_id": row["doc_id"],
            "source": row["source"],
            "path": row["path"],
            "text": row["text"],
            "tok_start": int(row["tok_start"]),
            "tok_end": int(row["tok_end"]),
        }
