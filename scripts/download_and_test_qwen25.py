import os
from huggingface_hub import hf_hub_download, login
from llama_cpp import Llama

REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"  # ~491 MB, good for CPU tests
LOCAL_DIR = "../models"

# Optional: use your HF token if needed
# token = os.getenv("HF_TOKEN")
# if token: login(token=token)

print("Downloading model…")
model_path = hf_hub_download(
    repo_id=REPO,
    filename=FILENAME,
    local_dir=LOCAL_DIR
)
print("Saved to:", model_path)

print("Loading with llama_cpp…")
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=os.cpu_count() or 8)

print("Running a tiny smoke test…")
out = llm("Q: 2+2?\nA:", max_tokens=16, temperature=0)
print("Output:", out["choices"][0]["text"].strip())
