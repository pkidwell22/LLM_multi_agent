# rag/embeddings.py
from typing import List
import os
import numpy as np
import torch
import importlib

# Make sure transformers doesn't try TF at all
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

def _auto_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load_ST():
    # Lazy, targeted import avoids pulling CrossEncoder and TF-heavy bits
    return importlib.import_module("sentence_transformers.SentenceTransformer").SentenceTransformer

def load_embedder(model_name: str, device: str = "auto"):
    SentenceTransformer = _load_ST()
    dev = _auto_device(device)
    return SentenceTransformer(model_name, device=dev)

def encode_texts(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
