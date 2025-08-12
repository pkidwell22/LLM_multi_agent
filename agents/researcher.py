# agents/researcher.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from .state import AgentState, Evidence

# We expect an object with .search(query, top_k, shards=None, per_shard_k=None) -> List[dict]
def run(state: AgentState, retriever: Any, reranker_fn, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
    # Wide recall, then CE rerank to top_k
    per_shard_k = state.per_shard_k or 50
    try:
        hits: List[Dict] = retriever.search(
            state.query,
            top_k=per_shard_k * (len(state.shards) if state.shards else 1),
            shards=state.shards,
            per_shard_k=per_shard_k,
        )
    except TypeError:
        # Fallback for older retriever signature
        hits = retriever.search(state.query, top_k=per_shard_k)

    if not hits:
        state.add_trace("researcher", "retrieve", "no-hits")
        return

    # Cross-encoder rerank (accepts top_k/k aliases)
    reranked = reranker_fn(
        query=state.query,
        hits=hits,
        model_name=cross_encoder_model,
        top_k=state.top_k,
    )

    state.evidence = [
        Evidence(
            chunk_id=h.get("chunk_id"),
            text=h.get("text",""),
            score=float(h.get("score", 0.0)),
            shard=h.get("shard"),
            meta={k:v for k,v in h.items() if k not in ("text","score","chunk_id","shard")},
            ce_score=h.get("ce_score"),
        )
        for h in reranked[: state.top_k]
    ]
    state.add_trace("researcher", "rerank", "select-evidence",
                    selected=len(state.evidence),
                    shards=list({e.shard for e in state.evidence if e.shard}))
