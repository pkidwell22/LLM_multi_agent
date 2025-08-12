# agents/llm_registry.py
from __future__ import annotations
import os, sys, pathlib, threading, importlib.util
from typing import Dict, Any, List

def _ensure_llama_dlls() -> None:
    """On Windows, add llama_cpp/lib to DLL search path *before* importing Llama."""
    if os.name != "nt":
        return
    candidates: List[pathlib.Path] = []
    spec = importlib.util.find_spec("llama_cpp")
    if spec and spec.origin:
        candidates.append(pathlib.Path(spec.origin).parent / "lib")
    root = pathlib.Path(__file__).resolve().parents[1]
    for base in (root, pathlib.Path(sys.prefix), pathlib.Path(sys.base_prefix)):
        candidates.append(base / "Lib" / "site-packages" / "llama_cpp" / "lib")
    for d in candidates:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                pass
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

class LLMClient:
    def chat(self, messages: List[dict], **gen_kwargs) -> str: ...
    def warmup(self) -> None: ...

class LlamaCppClient(LLMClient):
    def __init__(self, spec: Dict[str, Any]):
        _ensure_llama_dlls()
        from llama_cpp import Llama  # import after DLL shim

        cfg = dict(spec)
        model_path = cfg.pop("model_path")  # don’t pass twice
        cfg.pop("backend", None)

        cfg.setdefault("n_ctx", 4096)
        cfg.setdefault("n_batch", 512)
        cfg.setdefault("seed", 0)
        cfg.setdefault("verbose", False)

        self.llm = Llama(model_path=model_path, **cfg)
        self.defaults = dict(temperature=0.2, max_tokens=512)

    def chat(self, messages: List[dict], **gen_kwargs) -> str:
        kw = {**self.defaults, **gen_kwargs}
        out = self.llm.create_chat_completion(messages=messages, **kw)
        return out["choices"][0]["message"]["content"]

    def warmup(self) -> None:
        try:
            _ = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1, temperature=0.0
            )
        except Exception:
            pass

class LLMRegistry:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.specs = cfg.get("llms", {})
        self.roles = cfg.get("llm_roles", {})
        self.instances: Dict[str, LLMClient] = {}
        self.lock = threading.Lock()

    def _build(self, name: str) -> LLMClient:
        spec = self.specs[name]
        backend = (spec.get("backend") or "llamacpp").lower()
        if backend == "llamacpp":
            return LlamaCppClient(spec)
        raise ValueError(f"Unsupported backend: {backend}")

    def get(self, name: str) -> LLMClient:
        with self.lock:
            if name not in self.instances:
                self.instances[name] = self._build(name)
            return self.instances[name]

    def use_role(self, role: str) -> LLMClient:
        name = self.roles.get(role) or self.roles.get("default") or next(iter(self.specs))
        return self.get(name)

def build_registry(cfg: Dict[str, Any]) -> LLMRegistry:
    return LLMRegistry(cfg)
