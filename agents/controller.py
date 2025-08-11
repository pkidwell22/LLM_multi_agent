from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Dict, Any, List

from agents.llm import chat_raw, load_llama_pair, _load as _load_single
from agents.graph import build_pipeline, init_services
import yaml
from pathlib import Path

TOOL_SPEC = """
You are a planner. Choose the next ACTION as compact JSON with one of:
- {"tool":"RAG_QUERY", "query":"...", "top_k": 10}
- {"tool":"STYLE", "tone":"plain|formal|academic|concise", "text":"..."}
- {"tool":"SUMMARIZE", "max_words": 120, "text":"..."}
- {"tool":"CRITIC", "text":"...", "criteria":"citations intact, no new claims"}
- {"tool":"FINAL", "text":"..."}  # use when ready to answer the user

Rules:
- Prefer RAG_QUERY first unless the answer is trivially known from prior steps.
- STYLE may only rephrase; it MUST NOT remove or change citation tags like [doc_id pX ¶Y].
- SUMMARIZE condenses context for yourself; do not return it as final.
- CRITIC flags issues; if problems are found, request another RAG_QUERY or adjust via STYLE.
- Always return ONLY one compact JSON object, nothing else.
"""

_json_re = re.compile(r"\{.*\}", re.S)

def _extract_json(text: str) -> Dict[str, Any] | None:
    m = _json_re.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

@dataclass
class Orchestrator:
    cfg: Dict[str, Any]
    controller: Any
    services: Dict[str, Any]
    pipeline: Any
    style_llm: Any
    main_llm: Any

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]):
        main_llm, style_llm = load_llama_pair(cfg)
        controller = _load_single(
            cfg["controller_llama_cpp"]["model_path"],
            cfg["controller_llama_cpp"].get("n_ctx", 4096),
            cfg["controller_llama_cpp"].get("n_gpu_layers", -1),
            cfg["controller_llama_cpp"].get("n_batch", 256),
        )
        services = init_services(cfg)
        pipeline = build_pipeline(cfg).compile()
        return cls(cfg, controller, services, pipeline, style_llm, main_llm)

    def tool_rag_query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        state = {"query": query, "_services": self.services}
        out = self.pipeline.invoke(state)
        return {"answer": out["answer"], "citations": out["citations"]}

    def tool_style(self, text: str, tone: str = "plain") -> str:
        tone_map = {
            "plain": "Clear, neutral, precise.",
            "formal": "Formally academic with smooth transitions.",
            "academic": "Rigorous academic prose; preserve claims and citations.",
            "concise": "Brief and crisp; no verbosity.",
        }
        system = (
            f"You are a stylist. Rewrite for {tone_map.get(tone,'clear style')} "
            "WITHOUT changing facts or any [doc_id pX ¶Y] tags. Do not invent content."
        )
        return chat_raw(self.style_llm, system, text, max_tokens=768, temperature=0.55)

    def tool_summarize(self, text: str, max_words: int = 120) -> str:
        system = f"Summarize to <= {max_words} words. Keep key claims. No new info."
        return chat_raw(self.main_llm, system, text, max_tokens=400, temperature=0.2)

    def tool_critic(self, text: str, criteria: str) -> str:
        system = (
            "You are a strict critic. Check the text against the criteria. "
            "Return a one-paragraph diagnosis, then either 'OK' or 'REVISE' on a new line."
        )
        return chat_raw(self.main_llm, system, text + f"\n\nCriteria: {criteria}", max_tokens=300, temperature=0.0)

    def run(self, user_query: str, max_steps: int = 6) -> Dict[str, Any]:
        scratch: List[Dict[str, Any]] = []
        last_answer = None

        for step in range(max_steps):
            context_for_controller = "\n\n".join(
                [f"Step {i+1} -> {json.dumps(ev)[:1200]}" for i, ev in enumerate(scratch)]
            )
            sys = (
                "Plan tool use to answer the user's question with citations. "
                "Think briefly and output ONE JSON action per the schema."
            )
            user = f"USER QUESTION:\n{user_query}\n\nRECENT EVENTS:\n{context_for_controller}\n\n{TOOL_SPEC}"
            action_text = chat_raw(self.controller, sys, user, max_tokens=256, temperature=0.2)
            action = _extract_json(action_text) or {"tool": "RAG_QUERY", "query": user_query, "top_k": 10}

            if action["tool"] == "RAG_QUERY":
                res = self.tool_rag_query(action.get("query", user_query), int(action.get("top_k", 10)))
                scratch.append({"RAG_QUERY": action, "RESULT": res})
                last_answer = res
            elif action["tool"] == "STYLE":
                styled = self.tool_style(action.get("text", ""), action.get("tone", "plain"))
                scratch.append({"STYLE": action, "RESULT": styled[:400]})
                last_answer = {"answer": styled, "citations": last_answer.get("citations") if last_answer else []}
            elif action["tool"] == "SUMMARIZE":
                sm = self.tool_summarize(action.get("text", ""), int(action.get("max_words", 120)))
                scratch.append({"SUMMARIZE": action, "RESULT": sm})
            elif action["tool"] == "CRITIC":
                diag = self.tool_critic(action.get("text", ""), action.get("criteria", ""))
                scratch.append({"CRITIC": action, "RESULT": diag})
                if "REVISE" in diag:
                    continue
            elif action["tool"] == "FINAL":
                return {"answer": action.get("text", ""), "citations": last_answer.get("citations", [])}

        if last_answer:
            return last_answer
        return {"answer": "No answer produced.", "citations": []}
