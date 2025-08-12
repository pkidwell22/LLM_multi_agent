# agents/state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Evidence:
    chunk_id: Any
    text: str
    score: float
    shard: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    ce_score: Optional[float] = None

@dataclass
class TraceStep:
    agent: str
    thought: str
    action: str
    observations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    query: str
    plan: List[str] = field(default_factory=list)
    messages: List[Dict[str, str]] = field(default_factory=list)  # chat-like: {"role","content"}
    evidence: List[Evidence] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    step: int = 0
    max_steps: int = 4
    shards: Optional[List[str]] = None
    per_shard_k: int = 50
    top_k: int = 6
    trace: List[TraceStep] = field(default_factory=list)

    def add_trace(self, agent: str, thought: str, action: str, **obs):
        self.trace.append(TraceStep(agent=agent, thought=thought, action=action, observations=obs))
