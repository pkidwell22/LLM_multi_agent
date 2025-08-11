import os, pathlib

# Make sure DLLs can be found (Windows)
libdir = pathlib.Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages" / "llama_cpp" / "lib"
if libdir.exists():
    os.add_dll_directory(str(libdir))
for p in (
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
):
    if os.path.isdir(p):
        os.add_dll_directory(p)

from llama_cpp import Llama
import llama_cpp as lc

print("gpu_offload_supported:", getattr(lc, "llama_supports_gpu_offload", lambda: "n/a")())

# Use your local model (already present)
model_path = r"../models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
print(f"Model path: {model_path}")

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=-1,   # push as much as possible onto the GPU
    n_batch=512,
    verbose=True,
)

out = llm("Q: 2+2?\nA:", max_tokens=16, temperature=0)
print("Model output:", out["choices"][0]["text"].strip())
