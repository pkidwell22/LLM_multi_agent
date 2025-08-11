# scripts/test_llamacpp_cuda.py
import yaml
from llama_cpp import Llama
from pathlib import Path

# Ensure LLAMA logs show CUDA init
import os
os.environ["LLAMA_LOG_LEVEL"] = "INFO"

# Load llama_cpp config from settings.yaml
root_dir = Path(__file__).resolve().parent.parent
settings_path = Path(r"D:\PythonProject\LLM_Multi_Agent\settings.yaml")
with open(settings_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["llama_cpp"]

print(f"Loaded llama_cpp config from {settings_path}:")
for k, v in cfg.items():
    print(f"  {k}: {v}")

# Instantiate Llama with verbose logs
llm = Llama(**cfg, verbose=True)

# Run a quick prompt
resp = llm("Hello from CUDA test", max_tokens=16)
print("\n=== Model output ===")
print(resp)
