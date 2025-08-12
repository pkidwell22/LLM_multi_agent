# rag/index.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import glob

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None


@dataclass
class ShardSpec:
    name: str
    index_path: Path
    meta_path: Path


class ShardedFaiss:
    """
    Robust FAISS shard loader with optional GPU.

    Supported config shapes (examples):

      faiss:
        index_dir: rag_store/index
        use_gpu: true
        gpu_device: 0
        shards: []                             # auto-discover under index_dir

      faiss:
        index_dir: rag_store/index
        use_gpu: true
        shards:
          - "rag_store/embeddings/gutenberg"   # directory string
          - { path: "rag_store/index" }        # directory dict (auto-discover inside)
          - { name: gutenberg                  # fully explicit files
              index_path: rag_store/index/index_shard_000.faiss
              meta_path:  rag_store/index/meta_shard_000.parquet }
          - { name: gutenberg
              path: rag_store/index            # directory + explicit filenames
              index_file: index_shard_000.faiss
              meta_file:  meta_shard_000.parquet }
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        # Default to rag_store/index; discovery will also look in embeddings/* if needed
        self.index_dir = Path(self.cfg.get("index_dir", "rag_store/index"))
        self.use_gpu = bool(self.cfg.get("use_gpu", False))
        self.gpu_device = int(self.cfg.get("gpu_device", 0))
        raw_specs = self.cfg.get("shards", [])

        self.shards: List[ShardSpec] = self._normalize_specs(raw_specs)
        self._indexes: List[Tuple[str, Any]] = []  # (shard_name, faiss.Index)

    # --------------------------- public API ---------------------------

    def load(self):
        """Load all shards, moving to GPU if requested and available."""
        if faiss is None:
            raise RuntimeError(
                "faiss library not available (install faiss-gpu on CUDA machines or faiss-cpu)."
            )

        self._indexes.clear()

        # Setup GPU resources if requested and available
        res = None
        gpu_ready = False
        if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                gpu_ready = True
                print(f"[faiss] GPU resources ready on device {self.gpu_device}")
            except Exception as e:
                print(f"[faiss] GPU init failed ({e}) — falling back to CPU.")
                gpu_ready = False

        # If no shards discovered yet (e.g., index_dir empty), also try common embeddings layout
        if not self.shards:
            alt = Path("rag_store/embeddings")
            if alt.exists():
                # discover one level deep (e.g., embeddings/gutenberg)
                for p in sorted(alt.glob("*/")):
                    self.shards.extend(self._discover_dir(p, name_hint=p.name))

        for spec in self.shards:
            if not spec.index_path.exists():
                raise FileNotFoundError(f"FAISS file not found: {spec.index_path}")
            if not spec.meta_path.exists():
                raise FileNotFoundError(f"Meta parquet not found: {spec.meta_path}")

            idx = faiss.read_index(str(spec.index_path))

            # Move to GPU if requested and possible
            if gpu_ready:
                try:
                    idx = faiss.index_cpu_to_gpu(res, self.gpu_device, idx)
                    print(f"[faiss] loaded {spec.name} on GPU:{self.gpu_device}")
                except Exception as e:
                    print(f"[faiss] index_cpu_to_gpu failed for {spec.name} ({e}); keeping on CPU.")
                    print(f"[faiss] loaded {spec.name} on CPU")
            else:
                print(f"[faiss] loaded {spec.name} on CPU")

            self._indexes.append((spec.name, idx))

        if not self._indexes:
            print(f"[faiss] WARNING: no shards loaded from {self.index_dir}")

        return self  # allow chaining

    def search(self, query_vecs, top_k: int = 12) -> List[Dict[str, Any]]:
        """
        Search top_k neighbors across all shards for each input vector.

        Parameters
        ----------
        query_vecs : np.ndarray[float32] of shape (n, d)
            Precomputed embeddings. Ensure dtype=float32 & row-major.
        top_k : int
            Number of results to return after merging shards.

        Returns
        -------
        List[Dict]:
          [
            {"meta": {"shard": <name>, "row_id": <int>}, "score": <float>},
            ...
          ]
        """
        if not self._indexes:
            self.load()

        all_scores: List[Tuple[str, int, float]] = []  # (shard_name, row_id, dist)

        for shard_name, index in self._indexes:
            D, I = index.search(query_vecs, top_k)
            # Collect valid results
            for row in range(D.shape[0]):
                for col in range(D.shape[1]):
                    ridx = int(I[row, col])
                    if ridx < 0:
                        continue
                    dist = float(D[row, col])
                    all_scores.append((shard_name, ridx, dist))

        # Lower distance is closer in L2; change sort key if you're using IP
        all_scores.sort(key=lambda t: t[2])

        out: List[Dict[str, Any]] = []
        for shard_name, ridx, dist in all_scores[:top_k]:
            out.append(
                {
                    "meta": {"shard": shard_name, "row_id": ridx},
                    "score": dist,
                }
            )
        return out

    def release(self):
        """Explicitly clear loaded FAISS indices."""
        self._indexes.clear()

    # Backward-compat for older code that expects .indexes
    @property
    def indexes(self) -> List[Tuple[str, Any]]:
        """
        Returns a list of (shard_name, faiss.Index) pairs.
        Kept for backward compatibility with existing pipeline code.
        """
        return list(self._indexes)

    # ------------------------ normalization helpers ------------------------

    def _normalize_specs(self, raw_specs: Any) -> List[ShardSpec]:
        """
        Turn mixed shapes into a clean list of ShardSpec.
        Discovery order inside a directory:
          - index_shard_*.faiss + meta_shard_*.parquet
          - index.faiss + meta.parquet
          - *.faiss + *.parquet (best-effort pairing)
        """
        specs: List[ShardSpec] = []

        # If no shards provided, discover under index_dir
        if not raw_specs:
            specs.extend(self._discover_dir(self.index_dir, name_hint=self.index_dir.name))
            return specs

        # Single string => directory
        if isinstance(raw_specs, str):
            specs.extend(self._discover_dir(Path(raw_specs)))
            return specs

        # List => normalize each item
        if isinstance(raw_specs, list):
            for i, item in enumerate(raw_specs):
                if isinstance(item, str):
                    specs.extend(self._discover_dir(Path(item), name_hint=f"shard_{i:03d}"))
                    continue

                if isinstance(item, dict):
                    name = item.get("name") or f"shard_{i:03d}"

                    # Fully explicit files
                    if "index_path" in item and "meta_path" in item:
                        specs.append(
                            ShardSpec(
                                name=name,
                                index_path=Path(item["index_path"]),
                                meta_path=Path(item["meta_path"]),
                            )
                        )
                        continue

                    # Directory + optional specific filenames
                    if "path" in item:
                        base = Path(item["path"])
                        index_file = item.get("index_file")
                        meta_file = item.get("meta_file")
                        if index_file and meta_file:
                            specs.append(
                                ShardSpec(
                                    name=name,
                                    index_path=base / index_file,
                                    meta_path=base / meta_file,
                                )
                            )
                        else:
                            specs.extend(self._discover_dir(base, name_hint=name))
                        continue

                # anything else → ignore gracefully
            return specs

        # Fallback: try to treat it as a directory path
        try:
            specs.extend(self._discover_dir(Path(str(raw_specs))))
        except Exception:
            pass
        return specs

    def _discover_dir(self, base: Path, name_hint: Optional[str] = None) -> List[ShardSpec]:
        base = base.resolve()
        if not base.exists():
            return []

        patterns = [
            ("index_shard_*.faiss", "meta_shard_*.parquet"),
            ("index.faiss", "meta.parquet"),
            ("*.faiss", "*.parquet"),
        ]

        pairs: List[ShardSpec] = []
        for faiss_glob, meta_glob in patterns:
            faiss_files = sorted(glob.glob(str(base / faiss_glob)))
            meta_files = sorted(glob.glob(str(base / meta_glob)))
            if not faiss_files or not meta_files:
                continue

            matched = self._pair_by_suffix(faiss_files, meta_files)
            if not matched:
                matched = list(zip(faiss_files, meta_files))

            for idx, (fi, mi) in enumerate(matched):
                nm = name_hint or base.name
                if len(matched) > 1:
                    nm = f"{nm}_{idx:03d}"
                pairs.append(ShardSpec(name=nm, index_path=Path(fi), meta_path=Path(mi)))

            if pairs:
                break

        return pairs

    @staticmethod
    def _pair_by_suffix(faiss_files: List[str], meta_files: List[str]) -> List[Tuple[str, str]]:
        """
        Pair files like:
          index_shard_000.faiss  <-> meta_shard_000.parquet
          index_abc.faiss        <-> meta_abc.parquet
        """

        def suf(p: str) -> str:
            stem = Path(p).stem  # e.g., "index_shard_000" or "index"
            for token in ("index_shard_", "index_", "index", "meta_shard_", "meta_", "meta"):
                stem = stem.replace(token, "")
            return stem

        faiss_map = {suf(p): p for p in faiss_files}
        meta_map = {suf(p): p for p in meta_files}
        out: List[Tuple[str, str]] = []
        for key, fpath in faiss_map.items():
            mpath = meta_map.get(key)
            if mpath:
                out.append((fpath, mpath))
        return out

    # ----------------------------- misc ------------------------------------

    def __repr__(self) -> str:
        return f"ShardedFaiss(shards={len(self.shards)}, loaded={len(self._indexes)}, use_gpu={self.use_gpu}, device={self.gpu_device})"
