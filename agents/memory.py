# agents/memory.py
from __future__ import annotations
from typing import List, Dict
from collections import defaultdict
import time

class MemoryStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._sessions: Dict[str, Dict] = defaultdict(dict)
        self._ttl = ttl_seconds

    def _now(self): return int(time.time())

    def upsert(self, session_id: str, messages: List[Dict[str, str]]):
        s = self._sessions[session_id]
        s.setdefault("messages", [])
        s["messages"].extend(messages)
        s["ts"] = self._now()

    def get(self, session_id: str) -> List[Dict[str, str]]:
        s = self._sessions.get(session_id)
        if not s: return []
        # TTL cleanup
        if self._now() - s.get("ts", 0) > self._ttl:
            self._sessions.pop(session_id, None)
            return []
        return s.get("messages", [])

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)
