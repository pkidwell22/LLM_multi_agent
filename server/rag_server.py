# server/rag_server.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Any, Dict

# Allow Intel/LLVM OpenMP to coexist (prevents crashes on Windows)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make repo importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import yaml
import pandas as pd

app = FastAPI(title="Literature RAG Server")
_pipeline = None
_services = None
_cfg: Dict[str, Any] = {}
_CHUNK_LOOKUP: Dict[str, Dict[str, Any]] = {}


class QueryIn(BaseModel):
    query: str
    top_k: Optional[int] = None
    shards: Optional[List[str]] = None  # e.g., ["gutenberg", "reading"]


@app.on_event("startup")
def _startup():
    # Lazy import here so nothing pulls in torch before llama-cpp init
    from agents.graph import build_pipeline, init_services

    global _pipeline, _services, _cfg, _CHUNK_LOOKUP
    root = Path(__file__).resolve().parents[1]
    _cfg = yaml.safe_load((root / "config" / "settings.yaml").read_text()) or {}

    # Build chunk_id -> {path, source} map to enrich citations
    chunks_path = root / "rag_store" / "chunks" / "chunks.parquet"
    if chunks_path.exists():
        cdf = pd.read_parquet(chunks_path, columns=["chunk_id", "path", "source"])
        _CHUNK_LOOKUP = cdf.set_index("chunk_id")[["path", "source"]].to_dict(orient="index")
        print(f"[diag] chunk map ready → {len(_CHUNK_LOOKUP):,} ids")
    else:
        print("[diag] no chunks.parquet found; source enrichment limited")

    # Initialize services (llama-cpp should load BEFORE any torch import)
    _services = init_services(_cfg)
    _pipeline = build_pipeline(_cfg).compile()
    print("[ok] pipeline ready")


@app.get("/health")
def health():
    return {"ok": True}


def _enrich_citation(c: Dict[str, Any]) -> Dict[str, Any]:
    chunk_id = c.get("chunk_id") or c.get("id") or c.get("chunkId")
    score = c.get("score") or c.get("distance") or c.get("faiss_score")
    shard = c.get("shard") or c.get("source")
    path = c.get("path")

    if chunk_id and (not path or not shard):
        info = _CHUNK_LOOKUP.get(chunk_id) or {}
        path = path or info.get("path")
        shard = shard or info.get("source")

    out = {"chunk_id": chunk_id, "path": path, "shard": shard, "score": score}
    if isinstance(c.get("text"), str):
        out["text"] = c["text"]
    return out


@app.post("/query")
def query(q: QueryIn):
    state = {"query": q.query, "_services": _services}
    if q.top_k is not None:
        state["top_k"] = int(q.top_k)
    if q.shards:
        state["shards"] = list(q.shards)

    out = _pipeline.invoke(state)

    answer = out.get("answer", "")
    raw = out.get("citations")
    if not isinstance(raw, list) or not raw:
        raw = out.get("sources") or []
    sources = [_enrich_citation(c if isinstance(c, dict) else {}) for c in raw]

    return {"answer": answer, "sources": sources, "citations": raw}


if __name__ == "__main__":
    uvicorn.run("server.rag_server:app", host="127.0.0.1", port=8000, reload=False)
