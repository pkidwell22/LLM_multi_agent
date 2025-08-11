# scripts/init_rag_store.py
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "rag_store"

# Create directories
for p in [
    DATA / "raw/books",
    DATA / "raw/jstor_zips",
    DATA / "raw/jstor",
    DATA / "parsed",
    DATA / "chunks",
    DATA / "embeddings/books",
    DATA / "embeddings/jstor",
    DATA / "catalog",
    ROOT / "rag"
]:
    p.mkdir(parents=True, exist_ok=True)

cfg = {
    "data_root": str(DATA),
    "embeddings": {
        "model": "BAAI/bge-base-en-v1.5",
        "batch_size": 64,
        "device": "auto"
    },
    "reranker": {
        "enabled": True,
        "model": "BAAI/bge-reranker-base"
    },
    "faiss": {
        "books": {
            "path": str(DATA / "embeddings/books"),
            "factory": "IVF4096,PQ64",
            "dim": 768
        },
        "jstor": {
            "path": str(DATA / "embeddings/jstor"),
            "factory": "IVF4096,PQ64",
            "dim": 768
        }
    },
    "ocr": {
        "enabled": True
    },
    "llama_cpp": {
        "model_path": "./models/qwen2.5-7b-instruct-q4_k_m.gguf",
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "n_batch": 512
    },
    "citations": {
        "paragraphs": True
    }
}

config_dir = ROOT / "config"
config_dir.mkdir(exist_ok=True)

with open(config_dir / "settings.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("[ok] rag_store scaffold and config/settings.yaml created.")
