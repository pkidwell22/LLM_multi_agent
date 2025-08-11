# scripts/ingest_pdfs_folder.py
from pathlib import Path
import argparse
import re
import subprocess
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import fitz  # PyMuPDF
from ebooklib import epub
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
DATA = ROOT / "rag_store"
PARSED = DATA / "parsed"
PARSED.mkdir(parents=True, exist_ok=True)
OUT = PARSED / "parsed_docs.parquet"

def _ocr_pdf(pdf_path: Path) -> str:
    temp_prefix = pdf_path.with_suffix("")
    subprocess.run(f'pdftoppm -r 300 "{pdf_path}" "{temp_prefix}" -png', shell=True, check=True)
    texts = []
    for png in sorted(pdf_path.parent.glob(temp_prefix.name + "-*.png")):
        subprocess.run(f'tesseract "{png}" "{png}" -l eng --oem 1 --psm 1', shell=True, check=True)
        txt_file = png.with_suffix(".png.txt")
        texts.append(txt_file.read_text(encoding="utf-8", errors="ignore"))
        png.unlink(missing_ok=True)
        txt_file.unlink(missing_ok=True)
    return "\n".join(texts)

def _parse_pdf(p: Path, allow_ocr=True) -> str:
    doc = fitz.open(p)
    pages, needs_ocr = [], False
    for page in doc:
        t = page.get_text("text")
        if not t.strip():
            needs_ocr = True
            break
        pages.append(t)
    doc.close()
    if needs_ocr and allow_ocr and CFG.get("ocr", {}).get("enabled", True):
        return _ocr_pdf(p)
    return "\n".join(pages)

def _parse_epub(p: Path) -> str:
    book = epub.read_epub(str(p))
    html = []
    for item in book.get_items():
        if item.get_type() == 9:
            try:
                html.append(item.get_body_content().decode("utf-8", errors="ignore"))
            except Exception:
                pass
    txt = re.sub(r"<[^>]+>", " ", " ".join(html))
    return re.sub(r"\s+", " ", txt).strip()

def _parse_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _append_rows(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    tbl = pa.Table.from_pandas(df)
    if OUT.exists():
        existing = pq.read_table(OUT)
        concat = pa.concat_tables([existing, tbl])
        pq.write_table(concat, OUT)
    else:
        pq.write_table(tbl, OUT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with PDFs/TXTs/EPUBs")
    ap.add_argument("--source_key", required=True, help="value to set in 'source' (must match a faiss shard key)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"missing folder: {src}")

    rows = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in [".pdf", ".txt", ".epub"]:
            continue
        try:
            if ext == ".pdf":
                text = _parse_pdf(p, allow_ocr=True)
            elif ext == ".epub":
                text = _parse_epub(p)
            else:
                text = _parse_txt(p)
        except Exception as e:
            print(f"[warn] failed {p.name}: {e}")
            continue
        if not text or not text.strip():
            continue
        rows.append({
            "doc_id": p.stem,
            "source": args.source_key,   # <-- must match FAISS shard key
            "path": str(p),
            "text": text
        })
        # flush in batches to keep memory low
        if len(rows) >= 1000:
            _append_rows(rows)
            rows.clear()
    if rows:
        _append_rows(rows)
    print(f"[ok] ingested from {src} with source='{args.source_key}' into {OUT}")

if __name__ == "__main__":
    main()
