# agents/critic.py
from __future__ import annotations
from .state import AgentState

def run(state: AgentState, min_items: int = 3, min_span_chars: int = 400) -> bool:
    """Return True if evidence is sufficient; False to ask for another research hop."""
    if not state.evidence:
        state.add_trace("critic", "check", "insufficient", reason="no-evidence")
        return False
    total_chars = sum(len(e.text) for e in state.evidence)
    ok = (len(state.evidence) >= min_items) and (total_chars >= min_span_chars)
    state.add_trace("critic", "check", "sufficient" if ok else "insufficient",
                    items=len(state.evidence), chars=total_chars)
    return ok
