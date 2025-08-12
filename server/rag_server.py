# server/rag_server.py
from __future__ import annotations

import os, sys, time, uuid, traceback, threading, inspect, re
from pathlib import Path
from typing import List, Optional, Any, Dict
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import yaml
import pandas as pd

app = FastAPI(title="Literature RAG Multi-Agent Server")

_cfg: Dict[str, Any] = {}
_services: Dict[str, Any] | None = None
_pipeline = None

_CHUNK_LOOKUP: Dict[str, Dict[str, Any]] = {}  # chunk_id -> {path, source}
_CHUNKS_PARQUET = ROOT / "rag_store" / "chunks" / "chunks.parquet"
_CHUNK_TEXT_CACHE: Dict[str, str] = {}

_MEM: Dict[str, Dict[str, Any]] = {}
_MEM_LOCK = threading.Lock()

_REFUSAL_PATTERNS = (
    "no evidence provided",
    "there is no evidence",
    "unable to provide a response",
    "insufficient evidence",
    "cannot answer based on the provided evidence",
    "i cannot answer",
    "i'm unable to provide",
)

def _looks_like_refusal(text: str) -> bool:
    if not isinstance(text, str):
        return True
    t = text.strip().lower()
    if len(t) < 40:
        return True
    return any(p in t for p in _REFUSAL_PATTERNS)

class QueryIn(BaseModel):
    query: str
    top_k: Optional[int] = None
    per_shard_k: Optional[int] = None
    shards: Optional[List[str]] = None

class AgentRunIn(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_steps: int = 3
    top_k: Optional[int] = None
    per_shard_k: Optional[int] = None
    shards: Optional[List[str]] = None
    rerank: Optional[bool] = True

class ChatIn(BaseModel):
    session_id: Optional[str] = None
    message: str
    top_k: Optional[int] = None
    per_shard_k: Optional[int] = None
    shards: Optional[List[str]] = None
    max_steps: int = 3

class ChatResetIn(BaseModel):
    session_id: str = Field(..., description="Session to clear")

class AgentState(SimpleNamespace):
    def __init__(self, sid: str | None, **kwargs):
        super().__init__(**kwargs)
        self._sid = sid
        self._traces: List[Dict[str, Any]] = []

    def add_trace(self, step: str, **payload):
        rec = {"step": step, "time": time.time(), **payload}
        self._traces.append(rec)
        if self._sid:
            _trace_add(self._sid, step, payload)

    def get_traces(self) -> List[Dict[str, Any]]:
        return list(self._traces)

@app.on_event("startup")
def _startup():
    from agents.graph import build_pipeline, init_services
    global _cfg, _services, _pipeline, _CHUNK_LOOKUP

    cfg_path = ROOT / "config" / "settings.yaml"
    _cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    if _CHUNKS_PARQUET.exists():
        try:
            cdf = pd.read_parquet(_CHUNKS_PARQUET, columns=["chunk_id", "path", "source"])
            _CHUNK_LOOKUP = cdf.set_index("chunk_id")[["path", "source"]].to_dict(orient="index")
            print(f"[diag] chunk map ready → {len(_CHUNK_LOOKUP):,} ids")
        except Exception:
            print("[diag] chunks.parquet present but could not be read; continuing without enrichment")
    else:
        print("[diag] no chunks.parquet found; source enrichment limited")

    _services = init_services(_cfg)
    _pipeline = build_pipeline(_cfg).compile()

    try:
        _ensure_llm_ready()
    except Exception:
        print("[warn] LLM warmup skipped:", traceback.format_exc())

    print("[ok] pipeline ready")

def _ensure_llm_ready():
    llm = (_services or {}).get("llm")
    if not llm:
        return
    if hasattr(llm, "create_chat_completion"):
        try:
            llm.create_chat_completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1, temperature=0.0, stream=False,
            )
        except Exception:
            pass

def _session(session_id: str | None) -> str:
    sid = session_id or uuid.uuid4().hex
    with _MEM_LOCK:
        _MEM.setdefault(sid, {"messages": [], "traces": [], "meta": {"created_at": time.time()}})
    return sid

def _mem_add(sid: str, role: str, content: str):
    with _MEM_LOCK:
        _MEM.setdefault(sid, {"messages": [], "traces": [], "meta": {"created_at": time.time()}})
        _MEM[sid]["messages"].append({"role": role, "content": content, "ts": time.time()})

def _trace_add(sid: str, step: str, payload: Dict[str, Any]):
    with _MEM_LOCK:
        _MEM.setdefault(sid, {"messages": [], "traces": [], "meta": {"created_at": time.time()}})
        _MEM[sid]["traces"].append({"step": step, "time": time.time(), **payload})

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

# ---------- hydrate chunk text from disk using the span in chunk_id ----------
_span_re = re.compile(r"^(.*?):(\d+)-(\d+)$")

def _parse_span(chunk_id: str) -> tuple[int, int] | None:
    m = _span_re.match(chunk_id or "")
    if not m:
        return None
    try:
        return int(m.group(2)), int(m.group(3))
    except Exception:
        return None

def _read_span_text(path: str, start: int, end: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            s = f.read()
        start = max(0, min(start, len(s)))
        end = max(start, min(end, len(s)))
        pad = 220
        lo = max(0, start - pad)
        hi = min(len(s), end + pad)
        return s[lo:hi].strip()
    except Exception:
        return ""

def _hydrate_hits_text(hits: List[Dict[str, Any]]) -> None:
    """Ensure each hit has a 'text' field by reading its span from the source file."""
    for h in hits:
        if not isinstance(h, dict):
            continue
        if isinstance(h.get("text"), str) and h["text"].strip():
            continue
        cid = h.get("chunk_id") or h.get("id") or ""
        if not cid:
            continue
        if cid in _CHUNK_TEXT_CACHE:
            h["text"] = _CHUNK_TEXT_CACHE[cid]
            continue

        path = h.get("path") or (_CHUNK_LOOKUP.get(cid) or {}).get("path")
        span = _parse_span(cid)
        txt = ""
        if path and span:
            txt = _read_span_text(path, *span)
        elif _CHUNKS_PARQUET.exists():
            try:
                df = pd.read_parquet(_CHUNKS_PARQUET, columns=["chunk_id", "text"])
                row = df[df["chunk_id"] == cid]
                if len(row):
                    txt = str(row.iloc[0]["text"])
            except Exception:
                txt = ""

        if txt:
            h["text"] = txt
            if len(_CHUNK_TEXT_CACHE) > 5000:
                _CHUNK_TEXT_CACHE.clear()
            _CHUNK_TEXT_CACHE[cid] = txt

def _keyword_terms(q: str) -> list[tuple[str, int]]:
    """Return [(term_or_stem, weight), ...] for simple lexical scoring."""
    ql = (q or "").lower()
    terms = set()
    for t in re.findall(r"[a-z]{3,}", ql):
        terms.add(t)
    boosters = {
        "aristotle": 5, "aristotel": 5,
        "poetics": 5, "poetic": 4,
        "catharsis": 5, "katharsis": 5,
        "tragedy": 3, "tragic": 3,
        "pity": 2, "fear": 2,
        "purification": 2, "purgation": 2,
        "plato": 1, "republic": 1, "symposium": 1
    }
    out = []
    for t, w in boosters.items():
        if t in ql:
            out.append((t, w))
    for stem, w in [("aristotel", 5), ("poetic", 4), ("cathars", 5)]:
        out.append((stem, w))
    dedup = {}
    for t, w in out:
        dedup[t] = max(dedup.get(t, 0), w)
    return sorted(dedup.items(), key=lambda x: -x[1])

def _lexical_score(hit: dict, terms: list[tuple[str, int]]) -> float:
    path = (hit.get("path") or "").lower()
    text = (hit.get("text") or "").lower()
    title = Path(path).stem.lower()
    s = 0.0
    for t, w in terms:
        if not t:
            continue
        if t in path:  s += 1.5 * w
        if t in title: s += 1.2 * w
        if t in text:  s += 1.0 * w
    if any(t for t, _ in terms if t.startswith("aristotel")):
        if "plato" in (path + text): s -= 4.0
        if "symposium" in (path + text): s -= 2.5
        if "republic" in (path + text): s -= 2.0
    return s

def _lexical_rerank(query: str, hits: list[dict]) -> list[dict]:
    terms = _keyword_terms(query)
    scored = []
    for h in hits:
        sc = _lexical_score(h, terms)
        v = h.get("score") or 0.0
        scored.append((sc, float(v) if isinstance(v, (int, float)) else 0.0, h))
    scored.sort(key=lambda t: (-(t[0]), -(t[1])))
    return [h for _, __, h in scored]

def _seed_by_filename(query: str, limit: int = 12) -> list[dict]:
    """If retrieval misses, seed candidates where file path suggests relevance."""
    ql = (query or "").lower()
    wants_aristotle = ("aristotle" in ql) or ("aristotel" in ql) or ("poetic" in ql) or ("cathars" in ql)
    if not wants_aristotle or not _CHUNK_LOOKUP:
        return []
    seeds = []
    for cid, info in _CHUNK_LOOKUP.items():
        p = (info.get("path") or "").lower()
        if any(k in p for k in ("aristotle", "poetic")):
            seeds.append({"chunk_id": cid, "path": info.get("path"), "shard": info.get("source")})
            if len(seeds) >= limit:
                break
    _hydrate_hits_text(seeds)
    return seeds

def _dedup_hits(hits: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for h in hits:
        cid = h.get("chunk_id") or h.get("id")
        if cid and cid in seen:
            continue
        seen.add(cid)
        out.append(h)
    return out

def _call_llm(messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.2) -> str:
    llm = (_services or {}).get("llm")
    if not llm:
        return ""
    if hasattr(llm, "create_chat_completion"):
        res = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        try:
            return res["choices"][0]["message"]["content"]
        except Exception:
            return str(res)
    if hasattr(llm, "invoke"):
        return str(llm.invoke(messages))
    if hasattr(llm, "__call__"):
        return str(llm(messages))
    return ""

def _safe_import(module_name: str) -> Any | None:
    try:
        import importlib
        return importlib.import_module(module_name)
    except Exception:
        return None

def _maybe_call(module: str, func_names: list[str], *, sid: Optional[str] = None, **kwargs) -> Any | None:
    mod = _safe_import(module)
    if not mod:
        return None

    qval = None
    for alias in ("query", "question", "prompt", "message", "text", "q", "input"):
        if alias in kwargs and kwargs[alias] is not None:
            qval = kwargs[alias]
            break

    for fn_name in func_names:
        f = getattr(mod, fn_name, None)
        if not callable(f):
            continue

        sig = inspect.signature(f)
        accepted = list(sig.parameters.keys())
        kmap = dict(kwargs)

        if "state" in accepted and not isinstance(kmap.get("state"), AgentState):
            kmap["state"] = AgentState(
                sid=sid,
                query=qval,
                shards=kwargs.get("shards"),
                top_k=kwargs.get("top_k"),
                per_shard_k=kwargs.get("per_shard_k"),
                messages=kwargs.get("messages"),
                cfg=kwargs.get("cfg"),
                services=kwargs.get("services"),
            )

        for name in accepted:
            if name not in kmap:
                if name in ("query", "question", "prompt", "message", "text", "q", "input") and qval is not None:
                    kmap[name] = qval

        filtered = {k: v for k, v in kmap.items() if k in accepted}

        try:
            return f(**filtered)
        except TypeError:
            pass

        try:
            if len(accepted) == 1:
                if "state" in accepted and "state" in kmap:
                    return f(kmap["state"])
                if qval is not None:
                    return f(qval)
        except TypeError:
            pass

        try:
            return f()
        except TypeError:
            continue

    return None

@app.get("/health")
def health():
    ok_pipeline = _pipeline is not None
    ok_services = bool(_services)
    svc_names = list((_services or {}).keys())
    return {
        "ok": ok_pipeline and ok_services,
        "pipeline": ok_pipeline,
        "services": {"ok": ok_services, "keys": svc_names},
        "memory_sessions": len(_MEM),
        "chunks_indexed": len(_CHUNK_LOOKUP),
    }

@app.post("/query")
def query(q: QueryIn):
    state: Dict[str, Any] = {"query": q.query, "_services": _services}
    if q.top_k is not None:
        state["top_k"] = int(q.top_k)
    if q.per_shard_k is not None:
        state["per_shard_k"] = int(q.per_shard_k)
    if q.shards:
        state["shards"] = list(q.shards)

    try:
        out = _pipeline.invoke(state)
    except TypeError as e:
        if "unexpected keyword argument 'top_k'" in str(e) and "top_k" in state:
            k = state.pop("top_k")
            state["k"] = k
            out = _pipeline.invoke(state)
        else:
            print("[error] /query TypeError:\n", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"TypeError: {e}")
    except Exception as e:
        print("[error] /query pipeline failure:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PipelineError: {e.__class__.__name__}: {e}")

    answer = out.get("answer", "") or out.get("text", "")
    raw = out.get("citations") or out.get("sources") or []
    hits = [_enrich_citation(c if isinstance(c, dict) else {}) for c in raw]
    _hydrate_hits_text(hits)

    return {"answer": answer, "sources": hits, "citations": raw}

def _agent_plan(query: str, shards: Optional[List[str]] = None, *, sid: Optional[str] = None) -> Dict[str, Any]:
    plan = _maybe_call(
        "agents.planner",
        ["plan", "router", "route"],
        sid=sid,
        query=query,
        shards=shards,
        cfg=_cfg,
        services=_services,
        messages=None,
    )
    if plan:
        return plan
    hops = 2 if any(k in query.lower() for k in ("compare", "contrast", "synthesize", "evidence")) else 1
    return {"route": "rag", "hops": hops, "reason": "fallback: heuristic"}

def _agent_research(
    query: str,
    top_k: int | None,
    per_shard_k: int | None,
    shards: Optional[List[str]],
    *,
    sid: Optional[str] = None
) -> Dict[str, Any]:
    # Try custom researcher first
    r = _maybe_call(
        "agents.researcher",
        ["research", "gather", "retrieve_loop"],
        sid=sid,
        query=query, top_k=top_k, per_shard_k=per_shard_k, shards=shards,
        cfg=_cfg, services=_services, messages=None
    )
    if r:
        raw_hits = r.get("hits") or []
        hits = [_enrich_citation(h if isinstance(h, dict) else {}) for h in raw_hits]
        _hydrate_hits_text(hits)
        r["hits"] = hits
        return r

    # Fallback: call compiled pipeline once (and again with widened K if needed)
    def _once(tk, psk):
        s = {"query": query, "_services": _services}
        if tk is not None: s["top_k"] = int(tk)
        if psk is not None: s["per_shard_k"] = int(psk)
        if shards: s["shards"] = list(shards)
        out = _pipeline.invoke(s)
        raw = out.get("citations") or out.get("sources") or []
        hits_ = [_enrich_citation(h if isinstance(h, dict) else {}) for h in raw]
        _hydrate_hits_text(hits_)
        ctx = "\n\n".join([h.get("text", "") for h in hits_ if isinstance(h, dict)])
        return hits_, ctx

    try:
        hits, context = _once(top_k, per_shard_k)
        if not hits or len(context) < 400:
            tk2 = (top_k or 8) * 3
            psk2 = (per_shard_k or 4) * 2
            hits, context = _once(tk2, psk2)

        # If nothing obviously Aristotelian, seed by filename and dedup
        if not any(("aristotl" in (h.get("text", "") + h.get("path", "")).lower() or
                    "poetic"   in (h.get("text", "") + h.get("path", "")).lower())
                   for h in hits):
            seeds = _seed_by_filename(query, limit=16)
            hits = _dedup_hits(hits + seeds)

        # Lightweight lexical rerank to push “aristotle/poetics/catharsis” upward
        hits = _lexical_rerank(query, hits)
        cap = max(12, (top_k or 8))
        hits = hits[:cap]

        ctx = "\n\n".join([h.get("text", "") for h in hits if h.get("text")])[:8000]
        return {"hits": hits, "context": ctx, "notes": "fallback: pipeline.retrieve + lexical boost"}
    except Exception as e:
        # Never crash the agent—return an empty but well-formed result
        return {"hits": [], "context": "", "notes": f"retrieve error: {e}"}

def _agent_critic(query: str, draft: str, hits: List[Dict[str, Any]], *, sid: Optional[str] = None) -> Dict[str, Any]:
    res = _maybe_call(
        "agents.critic",
        ["critique", "verify", "check"],
        sid=sid,
        query=query, draft=draft, hits=hits, cfg=_cfg, services=_services
    )
    if res:
        return res
    return {"ok": bool(hits), "needs_more_evidence": not bool(hits), "issues": []}

def _agent_synthesize(query: str, hits: List[Dict[str, Any]], uncertainty_note: bool = True, *, sid: Optional[str] = None) -> Dict[str, Any]:
    res = _maybe_call(
        "agents.answerer",
        ["synthesize", "answer", "compose"],
        sid=sid,
        query=query, hits=hits, cfg=_cfg, services=_services
    )
    if isinstance(res, dict) and isinstance(res.get("answer"), str) and not _looks_like_refusal(res["answer"]):
        return res

    ctx_chunks = []
    for h in hits[:12]:
        t = (h.get("text") or "").strip()
        if t:
            ctx_chunks.append(t[:1600])
    context = "\n\n".join(ctx_chunks)[:8000]

    prompt = (
        "You are a careful research assistant. Using ONLY the evidence below, answer the user's question.\n"
        "Cite with [#] markers that map to the provided sources. If evidence is weak, say so.\n\n"
        f"Question:\n{query}\n\nEvidence:\n{context}\n\nAnswer:"
    )
    text = _call_llm(
        messages=[{"role": "system", "content": "Follow the instructions precisely."},
                  {"role": "user", "content": prompt}],
        max_tokens=384, temperature=0.2,
    )
    cites = [h for h in hits[:10]]
    if (uncertainty_note and not ctx_chunks) or _looks_like_refusal(text):
        text = "Based on the retrieved sources, there isn’t enough explicit evidence to answer confidently. Try broadening the query or raising top_k."
    return {"answer": text, "citations": cites}

def _run_multi_agent(q: AgentRunIn, sid: str) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = []

    plan = _agent_plan(q.query, q.shards, sid=sid)
    trace.append({"step": "plan", **plan})
    _trace_add(sid, "plan", plan)

    hits: List[Dict[str, Any]] = []
    max_steps = max(1, int(q.max_steps))
    for step in range(max_steps):
        research = _agent_research(q.query, q.top_k, q.per_shard_k, q.shards, sid=sid)
        raw_hits = research.get("hits") or []
        hits = [_enrich_citation(h if isinstance(h, dict) else {}) for h in raw_hits]
        _hydrate_hits_text(hits)
        trace.append({"step": f"research:{step+1}", "notes": research.get("notes", ""), "num_hits": len(hits)})
        _trace_add(sid, f"research:{step+1}", {"notes": research.get("notes", ""), "num_hits": len(hits)})

        synth = _agent_synthesize(q.query, hits, sid=sid)
        draft = synth.get("answer", "")
        trace.append({"step": "draft", "chars": len(draft)})
        _trace_add(sid, "draft", {"chars": len(draft)})

        critic = _agent_critic(q.query, draft, hits, sid=sid)
        trace.append({"step": "critic", **critic})
        _trace_add(sid, "critic", critic)

        if not critic.get("needs_more_evidence"):
            return {"answer": draft, "sources": synth.get("citations", []), "trace": trace}

    final = _agent_synthesize(q.query, hits, uncertainty_note=True, sid=sid)
    return {"answer": final.get("answer", ""), "sources": final.get("citations", []), "trace": trace}

@app.post("/agent/run")
def agent_run(q: AgentRunIn):
    try:
        sid = _session(q.session_id)
        out = _run_multi_agent(q, sid)
        return {"session_id": sid, **out}
    except Exception as e:
        print("[error] /agent/run failed:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"AgentError: {e.__class__.__name__}: {e}")

@app.post("/chat")
def chat(q: ChatIn):
    try:
        sid = _session(q.session_id)
        _mem_add(sid, "user", q.message)

        agent_req = AgentRunIn(
            query=q.message,
            session_id=sid,
            max_steps=q.max_steps,
            top_k=q.top_k,
            per_shard_k=q.per_shard_k,
            shards=q.shards,
        )
        out = _run_multi_agent(agent_req, sid)

        _mem_add(sid, "assistant", out.get("answer", ""))
        return {"session_id": sid, **out, "messages": _MEM.get(sid, {}).get("messages", [])}
    except Exception as e:
        print("[error] /chat failed:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"ChatError: {e.__class__.__name__}: {e}")

@app.get("/memory/{session_id}")
def memory_get(session_id: str):
    with _MEM_LOCK:
        return _MEM.get(session_id) or {"messages": [], "traces": [], "meta": {}}

@app.delete("/memory/{session_id}")
def memory_delete(session_id: str):
    with _MEM_LOCK:
        existed = session_id in _MEM
        _MEM.pop(session_id, None)
    return {"deleted": existed}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.rag_server:app", host="127.0.0.1", port=8000, reload=False)
