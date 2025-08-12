# agents/graph.py
from agents.prompts import build_prompt
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
from langgraph.graph import StateGraph, END

from rag.search import Retriever
from rag.rerank import Reranker
from rag.citations import CitationMapper

# NEW: registry
from agents.llm_registry import build_registry


def node_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    retriever: Retriever = state["_services"]["retriever"]
    per_shard_k = int(state.get("per_shard_k", 60))
    top_k = int(state.get("top_k", 12))
    results = retriever.retrieve(state["query"], per_shard_k=per_shard_k, top_k=top_k)
    return {**state, "retrieved": results}


def node_rerank(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = state["_services"]["cfg"]
    if not cfg.get("reranker", {}).get("enabled", False):
        return state
    reranker: Reranker = state["_services"]["reranker"]
    # accept both top_k and k (pipeline-safe)
    k = int(state.get("top_k") or state.get("k") or 10)
    top = reranker.rerank(state["query"], state["retrieved"], top_k=k)
    return {**state, "reranked": top}


def _ctx_block(cands: List[Dict], limit_chars: int = 15000) -> str:
    out, total = [], 0
    for i, c in enumerate(cands, start=1):
        t = (c.get("text") or "").strip()
        if not t:
            continue
        if total + len(t) > limit_chars:
            break
        out.append(f"[{i}] {t}")
        total += len(t)
    return "\n\n".join(out)

def node_compose(state: Dict[str, Any]):
    services = state["_services"]

    # Choose LLM by role
    role = state.get("target_role") or "rag_answer"
    llm = services["llms"].use_role(role)

    # Use retrieved/reranked context only for RAG-style roles
    use = state.get("reranked") or state.get("retrieved") or []
    context_block = _ctx_block(use) if role in ("rag_answer", "summarize") else ""

    # Build a role-specific prompt
    q = state["query"]
    prompt = build_prompt(role, q, context=context_block)

    # Generate
    answer = llm.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=services["cfg"].get("generation", {}).get("max_new_tokens", 256),
        temperature=services["cfg"].get("generation", {}).get("temperature", 0.2),
    )

    # Keep a small slice of context for citations step
    return {**state, "answer": answer, "used_context": use[:6]}


def node_cite(state):
    """
    Build a resilient citations array from whatever metadata we have.
    Works with:
      - rich meta (source, path, doc_id, tok_start/tok_end, chunk_id)
      - minimal meta from FAISS (shard, row_id)
    """
    mapper = state["_services"]["citer"]
    used = state.get("used_context") or state.get("retrieved") or []
    cites = []

    for c in used:
        meta = c.get("meta") or {}
        # Tolerant extraction
        source = meta.get("source") or meta.get("shard")
        path = meta.get("path")
        doc_id = meta.get("doc_id") or meta.get("chunk_id") or meta.get("id")
        chunk_id = meta.get("chunk_id") or doc_id
        tok_start = meta.get("tok_start")
        tok_end = meta.get("tok_end")
        text = c.get("text", "")

        # Only call mapper if we have enough to be meaningful
        cite_payload = {
            "source": source,
            "path": path,
            "doc_id": doc_id,
            "tok_start": tok_start,
            "tok_end": tok_end,
            "text": text,
        }
        cite = {}
        try:
            if hasattr(mapper, "cite") and (path or doc_id or chunk_id):
                # mapper.cite should be robust to missing fields; guard anyway
                cite = mapper.cite({k: v for k, v in cite_payload.items() if v is not None}) or {}
        except Exception:
            # Don't fail the whole request if mapper is strict
            cite = {}

        cites.append({
            "doc_id": doc_id,
            "page": cite.get("page"),
            "paragraph": cite.get("paragraph"),
            "chunk_id": chunk_id,
            "source": source,
            "path": path,
            "score": c.get("score") or c.get("distance"),
            "text": text if text else None,
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

    # NEW: multi-LLM registry
    registry = build_registry(cfg)
    # Back-compat: expose a default single-llm handle too
    default_role = (cfg.get("llm_roles") or {}).get("default", "answerer")
    default_llm = registry.use_role(default_role)

    return {
        "cfg": cfg,
        "retriever": retriever,
        "reranker": reranker,
        "citer": citer,
        "llms": registry,   # <- primary
        "llm": default_llm, # <- legacy code path
    }


# ---------- (keep your multi-agent helpers below if you use them) ----------
from .state import AgentState  # noqa: E402
from . import planner as _planner  # noqa: E402
from . import researcher as _researcher  # noqa: E402
from . import critic as _critic  # noqa: E402
from . import answerer as _answerer  # noqa: E402
from rag.rerank import rerank as _rerank  # noqa: E402

def build_multi_agent(services: Dict[str, Any], cfg: Any):
    """Return a callable(query, **params) -> {answer, citations, trace}"""
    retriever = services["retriever"]
    # For agents, keep using the default LLM (or extend to choose roles per agent)
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
            _researcher.run(state, retriever=retriever, reranker_fn=_rerank)
            if _critic.run(state):
                break

        result = _answerer.synthesize(state, llm=llm)
        out = {**result}
        if trace:
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
