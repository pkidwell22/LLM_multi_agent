import yaml
from llama_cpp import Llama
from pathlib import Path

# Build path relative to this script's directory
root_dir = Path(__file__).resolve().parent.parent  # project root
settings_path = root_dir / "settings.yaml"

with open(settings_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["llama_cpp"]

llm = Llama(**cfg)
print(llm("Hello", max_tokens=24))
