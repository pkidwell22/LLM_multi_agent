# rag/rerank.py
from __future__ import annotations

from functools import lru_cache
from typing import List, Dict, Tuple


@lru_cache(maxsize=4)
def _get_ce(model_name: str):
    """
    Lazy-load CrossEncoder so torch isn't imported until after llama-cpp initializes.
    """
    from sentence_transformers import CrossEncoder  # noqa: WPS433 (intentional lazy import)
    return CrossEncoder(model_name, trust_remote_code=True)


class Reranker:
    """
    Backward-compatible shim used by agents/graph.py:
      rr = Reranker(model_name="BAAI/bge-reranker-base", top_n=8)
      hits = rr(query, hits)  # or rr.rerank(query, hits)

    hits: list of dicts with at least {'text': str}. We pass through other keys
    like 'chunk_id', 'score', 'path', 'shard'.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = 8):
        self.model_name = model_name
        self.top_n = int(top_n)

    def _model(self):
        return _get_ce(self.model_name)

    def rerank(self, query: str, hits: List[Dict]) -> List[Dict]:
        if not hits:
            return []
        ce = self._model()
        pairs: List[Tuple[str, str]] = [(query, (h.get("text") or "")) for h in hits]
        scores = ce.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        scored: List[Dict] = []
        for h, s in zip(hits, scores):
            h = dict(h)
            h["ce_score"] = float(s)
            scored.append(h)
        scored.sort(key=lambda x: x["ce_score"], reverse=True)
        return scored[: self.top_n]

    __call__ = rerank  # allow rr(query, hits)


def rerank(query: str, hits: List[Dict], model_name: str, top_n: int) -> List[Dict]:
    """
    Functional helper for pipelines that prefer a function call.
    """
    return Reranker(model_name=model_name, top_n=top_n).rerank(query, hits)
