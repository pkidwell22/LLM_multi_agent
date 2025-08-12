from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.index import ShardedFaiss

class _EncoderSingleton:
    _model: Optional[SentenceTransformer] = None
    _name: Optional[str] = None

    @classmethod
    def get(cls, name: str) -> SentenceTransformer:
        if cls._model is None or cls._name != name:
            cls._model = SentenceTransformer(name, trust_remote_code=True)
            cls._name = name
        return cls._model

class Retriever:
    """
    Wrapper around FAISS shards + query encoder.
    API: retrieve(query, per_shard_k=60, top_k=12) -> List[Dict]
    """

    def __init__(self, cfg: Dict[str, Any], chunks_parquet_path: str):
        self.cfg = cfg
        self.index = ShardedFaiss(cfg["faiss"])
        self.index.load()

        emb_cfg = cfg.get("embeddings", {})
        model_name = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = _EncoderSingleton.get(model_name)
        self.normalize = bool(emb_cfg.get("normalize", True))

    def _encode_query(self, text: str) -> np.ndarray:
        vec = self.encoder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )[0]
        if vec.dtype != np.float32:
            vec = vec.astype("float32", copy=False)
        return vec

    def retrieve(
        self,
        query: str,
        per_shard_k: int = 60,
        top_k: int = 12,
        shards: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        per_shard_k: how many to fetch per shard from index (overfetch)
        top_k:       how many to return after sorting
        shards:      optional allowlist of shard names
        """
        qv = self._encode_query(query)

        # Alias per_shard_k → top_k for FAISS
        hits = self.index.search(qv, top_k=per_shard_k, shards=shards)

        # Sort by ascending distance
        hits_sorted = sorted(
            hits,
            key=lambda h: float(h.get("distance", h.get("score", 1e9))),
        )
        return hits_sorted[:top_k]
