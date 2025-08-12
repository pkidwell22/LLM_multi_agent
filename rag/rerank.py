
# rag/rerank.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from sentence_transformers import CrossEncoder

_DEFAULT_CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Keep CE under 512 wordpiece tokens; we also hard-clip text by chars as a belt-and-suspenders.
_CHAR_CLIP = 1500

class CEPool:
    _cache: dict[str, CrossEncoder] = {}

    @classmethod
    def get(cls, name: Optional[str]) -> CrossEncoder:
        key = name or _DEFAULT_CE
        m = cls._cache.get(key)
        if m is None:
            # max_length enforces tokenizer truncation; trust_remote_code to be safe with HF CE repos
            m = CrossEncoder(key, max_length=512, trust_remote_code=True)
            cls._cache[key] = m
        return m

def _shorten(text: str) -> str:
    if not text:
        return ""
    # quick char-level clip to avoid huge CE inputs (many CEs warn/error > 512 tokens)
    return text[:_CHAR_CLIP]

class Reranker:
    """
    Used by agents/graph.py
    Accepts aliases top_k/k, ignores stray model_name kwargs.
    """
    def __init__(self, model_name: Optional[str] = None, **_):
        self.model_name = model_name
        self._ce: Optional[CrossEncoder] = None

    def _get(self) -> CrossEncoder:
        if self._ce is None:
            self._ce = CEPool.get(self.model_name)
        return self._ce

    def rerank(
        self,
        query: str,
        hits: List[Dict],
        *args,
        top_n: Optional[int] = None,
        **kwargs
    ) -> List[Dict]:
        kwargs.pop("model_name", None)
        alias = kwargs.pop("top_k", kwargs.pop("k", None))
        if alias is not None:
            top_n = int(alias)
        if top_n is None:
            top_n = 10
        if not hits:
            return []

        ce = self._get()
        pairs: List[Tuple[str, str]] = [(query, _shorten(h.get("text") or "")) for h in hits]
        scores = ce.predict(pairs, convert_to_numpy=True, show_progress_bar=False)

        for h, s in zip(hits, scores):
            h["ce_score"] = float(s)

        hits.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
        return hits[:top_n]

# Back-compat free function
def rerank(
    query: str,
    hits: List[Dict],
    model_name: Optional[str] = None,
    top_n: Optional[int] = None,
    **kwargs
) -> List[Dict]:
    rr = Reranker(model_name=model_name)
    return rr.rerank(query, hits, top_n=top_n, **kwargs)
