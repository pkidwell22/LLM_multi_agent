# tools/download_gguf.py
from huggingface_hub import snapshot_download
import sys, os

if len(sys.argv) < 2:
    print("Usage: python tools/download_gguf.py <repo_id> [pattern]", flush=True)
    print('Example: python tools/download_gguf.py "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" "*Q4_K_M.gguf"')
    raise SystemExit(1)

repo_id = sys.argv[1]
pattern = sys.argv[2] if len(sys.argv) > 2 else "*.gguf"

target_dir = os.path.join("models", repo_id.split("/")[-1])
path = snapshot_download(
    repo_id=repo_id,
    allow_patterns=pattern,          # e.g. "*Q4_K_M.gguf"
    local_dir=target_dir,
    local_dir_use_symlinks=False
)
print("Downloaded to:", path)
