# scripts/ingest_zip.py
import hashlib
import json
import zipfile
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "rag_store" / "raw"
ZIPS = RAW / "jstor_zips"
PDF_OUT = RAW / "jstor"
PDF_OUT.mkdir(exist_ok=True, parents=True)
MANIFEST = ROOT / "rag_store" / "catalog" / "ingest_manifest.json"

def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()

def main():
    seen = set()
    if MANIFEST.exists():
        seen = set(json.loads(MANIFEST.read_text()))
    for zf in sorted(ZIPS.glob("*.zip")):
        with zipfile.ZipFile(zf, "r") as z:
            for name in tqdm(z.namelist(), desc=f"unzipping {zf.name}"):
                if not name.lower().endswith(".pdf"):
                    continue
                data = z.read(name)
                h = sha1_bytes(data)
                if h in seen:
                    continue
                (PDF_OUT / f"{h}.pdf").write_bytes(data)
                seen.add(h)
    MANIFEST.write_text(json.dumps(sorted(seen)))
    print("[ok] ingest complete:", len(seen), "unique PDFs")

if __name__ == "__main__":
    main()
