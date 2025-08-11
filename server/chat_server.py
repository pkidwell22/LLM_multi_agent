from pathlib import Path
import uvicorn, yaml
from fastapi import FastAPI
from pydantic import BaseModel
from agents.controller import Orchestrator

app = FastAPI(title="Do-It-All Multi-Agent Server")
_orch: Orchestrator | None = None

class ChatIn(BaseModel):
    query: str
    max_steps: int | None = 6

@app.on_event("startup")
def _startup():
    global _orch
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "settings.yaml").read_text())
    _orch = Orchestrator.from_config(cfg)
    print("[ok] orchestrator ready")

@app.post("/chat")
def chat(q: ChatIn):
    out = _orch.run(q.query, max_steps=q.max_steps or 6)
    return out

if __name__ == "__main__":
    uvicorn.run("server.chat_server:app", host="127.0.0.1", port=8010, reload=False)
