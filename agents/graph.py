from typing import Dict, Any, List
import yaml
from pathlib import Path
from langgraph.graph import StateGraph, END

from agents.llm import load_llama_cpp, chat
from rag.search import Retriever
from rag.rerank import Reranker
from rag.citations import CitationMapper

def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    retriever: Retriever = state["_services"]["retriever"]
    results = retriever.retrieve(state["query"], per_shard_k=60, top_k=12)
    return {**state, "retrieved": results}

def node_rerank(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = state["_services"]["cfg"]
    if not cfg.get("reranker", {}).get("enabled", False):
        return state
    reranker: Reranker = state["_services"]["reranker"]
    top = reranker.rerank(state["query"], state["retrieved"], top_k=10)
    return {**state, "reranked": top}

def _ctx_block(cands: List[Dict], limit_chars: int = 15000) -> str:
    out, total = [], 0
    for i, c in enumerate(cands, start=1):
        t = c["text"].strip()
        if total + len(t) > limit_chars: break
        out.append(f"[{i}] {t}"); total += len(t)
    return "\n\n".join(out)

def node_compose(state):
    llm = state["_services"]["llm"]
    use = state.get("reranked") or state["retrieved"]
    context_block = _ctx_block(use)

    system = (
        "You are a literature research assistant. Only use the provided CONTEXT. "
        "Do not invent citations or bracketed references. If context is insufficient, say so."
    )
    user = (
        f"QUESTION:\n{state['query']}\n\n"
        f"CONTEXT (numbered):\n{context_block}\n\n"
        f"Write a concise answer in plain prose (no [1], no brackets)."
    )
    answer = chat(llm, system, user)
    return {**state, "answer": answer, "used_context": use[:6]}

def node_cite(state):
    mapper = state["_services"]["citer"]
    used = state["used_context"]
    cites = []
    for c in used:
        meta = c["meta"]
        cite = mapper.cite({
            "source": meta["source"],
            "path": meta["path"],
            "doc_id": meta["doc_id"],   # <-- add this
            "tok_start": meta["tok_start"],
            "tok_end": meta["tok_end"],
            "text": c["text"],
        })
        cites.append({
            "doc_id": meta["doc_id"],
            "page": cite.get("page"),
            "paragraph": cite.get("paragraph"),
            "chunk_id": meta["chunk_id"],
            "source": meta["source"],
        })
    return {**state, "citations": cites}



def build_pipeline(cfg: dict) -> StateGraph:
    g = StateGraph(dict)
    g.add_node("retrieve", node_retrieve)
    g.add_node("rerank", node_rerank)
    g.add_node("compose", node_compose)
    g.add_node("cite", node_cite)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "compose")
    g.add_edge("compose", "cite")
    g.add_edge("cite", END)
    return g

def init_services(cfg: dict) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    chunks = root / "rag_store" / "chunks" / "chunks.parquet"
    retriever = Retriever(cfg, str(chunks))
    reranker = Reranker(cfg["reranker"]["model"]) if cfg.get("reranker", {}).get("enabled", False) else None
    citer = CitationMapper(cfg["embeddings"]["model"])
    llm = load_llama_cpp(cfg)
    return {"cfg": cfg, "retriever": retriever, "reranker": reranker, "citer": citer, "llm": llm}

# agents/graph.py (append)
from typing import Dict, Any, Optional, List
from .state import AgentState
from . import planner as _planner
from . import researcher as _researcher
from . import critic as _critic
from . import answerer as _answerer
from rag.rerank import rerank as _rerank

def build_multi_agent(services: Dict[str, Any], cfg: Any):
    """Return a callable(query, **params) -> {answer, citations, trace}"""
    retriever = services["retriever"]
    llm = services["llm"]

    def run(query: str,
            shards: Optional[List[str]] = None,
            top_k: int = 6,
            per_shard_k: int = 50,
            max_steps: int = 4,
            trace: bool = False) -> Dict[str, Any]:

        state = AgentState(
            query=query,
            shards=shards,
            top_k=top_k,
            per_shard_k=per_shard_k,
            max_steps=max_steps,
        )

        _planner.plan(state)

        for _ in range(state.max_steps):
            state.step += 1
            # research
            _researcher.run(state, retriever=retriever, reranker_fn=_rerank)
            # critic
            if _critic.run(state):
                break

        # answer
        result = _answerer.synthesize(state, llm=llm)
        out = {**result}
        if trace:
            # serialize trace to plain dicts
            out["trace"] = [
                {
                    "agent": t.agent,
                    "thought": t.thought,
                    "action": t.action,
                    "observations": t.observations,
                }
                for t in state.trace
            ]
        return out

    return run
