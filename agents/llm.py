# agents/llm.py
import os, pathlib
libdir = pathlib.Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages" / "llama_cpp" / "lib"
if libdir.exists():
    os.add_dll_directory(str(libdir))

from pathlib import Path
from llama_cpp import Llama

def load_llama_cpp(cfg: dict) -> Llama:
    mp = Path(cfg["llama_cpp"]["model_path"]).resolve()
    print(f"[diag] loading gguf: {mp}  exists={mp.exists()}")
    return Llama(
        model_path=str(mp),
        n_ctx=cfg["llama_cpp"].get("n_ctx", 8192),
        n_gpu_layers=cfg["llama_cpp"].get("n_gpu_layers", -1),
        n_batch=cfg["llama_cpp"].get("n_batch", 512),
        logits_all=False,
        verbose=True,                # <— turn on llama.cpp logs
        seed=cfg["llama_cpp"].get("seed", 42),
    )

def chat(llm: Llama, system: str, user: str) -> str:
    prompt = (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    out = llm(prompt=prompt, max_tokens=512, temperature=0.2, top_p=0.95, stop=["<|im_end|>"])
    return out["choices"][0]["text"].strip()
