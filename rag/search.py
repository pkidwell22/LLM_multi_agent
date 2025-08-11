# rag/search.py
from typing import List, Dict
import numpy as np
from rag.embeddings import load_embedder, encode_texts
from rag.index import ShardedFaiss
from rag.docstore import DocStore

class Retriever:
    def __init__(self, cfg: dict, chunks_parquet: str):
        self.embedder = load_embedder(cfg["embeddings"]["model"], cfg["embeddings"].get("device", "auto"))
        self.index = ShardedFaiss(cfg["faiss"])
        self.index.load()
        self.store = DocStore(chunks_parquet)

    def retrieve(self, query: str, per_shard_k: int = 60, top_k: int = 12) -> List[Dict]:
        qv = encode_texts(self.embedder, [query], batch_size=1)[0].astype(np.float32)
        hits = self.index.search(qv, per_shard_k=per_shard_k, top_k=per_shard_k * len(self.index.indexes))
        # unify + take global top_k by distance
        hits.sort(key=lambda x: x[2])  # distance asc
        out: List[Dict] = []
        for shard, chunk_id, dist in hits[:top_k]:
            meta = self.store.get_chunk(chunk_id)
            out.append({
                "text": meta["text"],
                "meta": {
                    "chunk_id": chunk_id,
                    "doc_id": meta["doc_id"],
                    "source": meta["source"],
                    "path": meta["path"],
                    "tok_start": meta["tok_start"],
                    "tok_end": meta["tok_end"],
                    "distance": float(dist),
                    "shard": shard,
                }
            })
        return out
