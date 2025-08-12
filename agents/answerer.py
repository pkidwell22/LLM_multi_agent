# agents/answerer.py
from __future__ import annotations
from typing import List, Dict
from .state import AgentState, Evidence

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer concisely using the Evidence sections. "
    "If the evidence is weak or irrelevant, say you are unsure. Include bracketed numeric citations "
    "like [1], [2] that refer to the Evidence items you used. Do not invent sources."
)

def _build_evidence_sections(evs: List[Evidence]) -> str:
    lines = []
    for i, e in enumerate(evs, start=1):
        src = e.meta.get("path") or e.meta.get("title") or (e.shard or "source")
        lines.append(f"[{i}] ({src})\n{e.text}\n")
    return "\n".join(lines)

def synthesize(state: AgentState, llm, max_tokens: int = 256, temperature: float = 0.2) -> Dict:
    evidence_md = _build_evidence_sections(state.evidence)
    user = (
        f"Question:\n{state.query}\n\n"
        f"Evidence (ordered, cite as [#]):\n{evidence_md}\n\n"
        "Write the answer in 2–5 sentences and include citations."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    state.add_trace("answerer", "compose", "llama.cpp.create_chat_completion",
                    max_tokens=max_tokens, temperature=temperature)
    out = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = out["choices"][0]["message"]["content"]

    # Build simple citations map (keeps order of first appearance)
    used = []
    for i in range(1, len(state.evidence)+1):
        tag = f"[{i}]"
        if tag in text:
            used.append(i)
    citations = []
    for i in used:
        ev = state.evidence[i-1]
        citations.append({
            "n": i,
            "shard": ev.shard,
            "path": ev.meta.get("path"),
            "title": ev.meta.get("title"),
            "score": ev.score,
            "ce_score": ev.ce_score,
            "chunk_id": ev.chunk_id,
        })
    state.citations = citations
    return {"answer": text, "citations": citations}
