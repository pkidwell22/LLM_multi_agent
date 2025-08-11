# rag/index.py
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import faiss
import pyarrow.parquet as pq

class ShardedFaiss:
    """Load and search multiple FAISS shards (e.g., books + jstor)."""

    def __init__(self, shards_cfg: Dict[str, Dict]):
        self.cfg = shards_cfg
        self.indexes: Dict[str, faiss.Index] = {}
        self.id_maps: Dict[str, np.ndarray] = {}

    def load(self) -> None:
        for name, spec in self.cfg.items():
            idx_path = Path(spec["path"]) / "index.faiss"
            meta_path = Path(spec["path"]) / "meta.parquet"
            if not idx_path.exists() or not meta_path.exists():
                continue
            self.indexes[name] = faiss.read_index(str(idx_path))
            meta = pq.read_table(str(meta_path)).to_pandas()
            self.id_maps[name] = meta["chunk_id"].to_numpy()
        if not self.indexes:
            raise FileNotFoundError("No FAISS shards found; build indexes first.")

    def search(self, query_vec: np.ndarray, per_shard_k: int = 50, top_k: int = 50) -> List[Tuple[str, str, float]]:
        results: List[Tuple[str, str, float]] = []
        q = query_vec[None, :].astype(np.float32)
        for shard, idx in self.indexes.items():
            D, I = idx.search(q, per_shard_k)
            ids = self.id_maps[shard][I[0]]
            for dist, cid in zip(D[0], ids):
                results.append((shard, str(cid), float(dist)))
        results.sort(key=lambda x: x[2])
        return results[:top_k]
