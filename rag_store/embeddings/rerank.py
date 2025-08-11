# rag/rerank.py
from __future__ import annotations
from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder

class CE:
    _model: CrossEncoder | None = None

    @classmethod
    def get(cls, name: str) -> CrossEncoder:
        if cls._model is None:
            cls._model = CrossEncoder(name, trust_remote_code=True)
        return cls._model

def rerank(query: str, hits: List[Dict], model_name: str, top_n: int) -> List[Dict]:
    """
    hits: [{"chunk_id":..., "text":..., "score": float, "shard": str, ...}, ...]
    Returns best 'top_n' by cross-encoder score (desc).
    """
    if not hits:
        return []
    ce = CE.get(model_name)
    pairs: List[Tuple[str,str]] = [(query, h.get("text") or "") for h in hits]
    ce_scores = ce.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    for h, s in zip(hits, ce_scores):
        h["ce_score"] = float(s)
    return sorted(hits, key=lambda x: x["ce_score"], reverse=True)[:top_n]
