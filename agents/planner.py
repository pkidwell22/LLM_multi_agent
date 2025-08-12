# agents/planner.py
from __future__ import annotations
from typing import List
from .state import AgentState

RAG_KEYWORDS = {"who","what","when","where","why","how","explain","summarize","define","compare"}

def plan(state: AgentState) -> List[str]:
    q = state.query.lower()
    # Minimal router (MVP): if question-like → RAG; else still RAG
    steps = ["research", "critic", "answer"]
    if any(w in q for w in RAG_KEYWORDS):
        state.plan = steps
    else:
        state.plan = steps
    state.add_trace("planner", "choose-pipeline", "set-plan", plan=state.plan)
    return state.plan
