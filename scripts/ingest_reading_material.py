# scripts/ingest_reading_material.py
"""
Scan your external Reading Material folder (PDF/TXT/EPUB), extract text,
and append to rag_store/parsed/parsed_docs.parquet with source="reading".

Usage (PowerShell):
  python scripts/ingest_reading_material.py
  # or to override path:
  python scripts/ingest_reading_material.py --path "D:\Some\Other\Folder" --max_docs 0

Requires: PyMuPDF; optional OCR if CFG['ocr']['enabled']=true + tesseract + pdftoppm on PATH.
"""
from __future__ import annotations
import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import fitz  # PyMuPDF
from ebooklib import epub
import yaml
import shutil
import pyarrow as pa, pyarrow.parquet as pq

HAVE_PDFTOPPM = shutil.which("pdftoppm") is not None

# --- update _ocr_pdf to guard for missing poppler ---
def _ocr_pdf(pdf_path: Path) -> str:
    if not HAVE_PDFTOPPM:
        # No Poppler → skip OCR gracefully
        print(f"[warn] OCR skipped (pdftoppm not found) for {pdf_path.name}")
        return ""
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

# --- update _parse_pdf so it only OCRs when (a) empty text AND (b) poppler exists ---
def _parse_pdf(pdf_path: Path, allow_ocr: bool = True) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    needs_ocr = False
    for page in doc:
        text = page.get_text("text")
        if not text.strip():
            needs_ocr = True
            break
        pages.append(text)
    doc.close()

    if needs_ocr and allow_ocr and HAVE_PDFTOPPM:
        try:
            return _ocr_pdf(pdf_path)
        except Exception as e:
            print(f"[warn] OCR failed for {pdf_path.name}: {e}")

    # If OCR not available or failed, return whatever we got (may be empty)
    return "\n".join(pages)

# --- add a small normalizer so 'year' is always string and schema matches when appending ---
FIXED_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("source", pa.string()),
    ("path", pa.string()),
    ("text", pa.string()),
    ("title", pa.string()),
    ("year", pa.string()),     # ← force string
    ("authors", pa.string()),
    ("venue", pa.string()),
])

def _normalize_rows(rows):
    if not rows:
        return pa.Table.from_arrays([pa.array([], type=field.type) for field in FIXED_SCHEMA], schema=FIXED_SCHEMA)
    df = pd.DataFrame(rows)
    # ensure all expected columns exist
    for col in ["title", "year", "authors", "venue"]:
        if col not in df.columns:
            df[col] = ""
    # force types to str (Arrow will validate)
    for col in ["doc_id","source","path","text","title","authors","venue"]:
        df[col] = df[col].fillna("").astype(str)
    df["year"] = df["year"].astype(str).fillna("")
    # column order
    df = df[["doc_id","source","path","text","title","year","authors","venue"]]
    return pa.Table.from_pandas(df, schema=FIXED_SCHEMA, preserve_index=False)

def _append_rows(rows):
    table = _normalize_rows(rows)
    if OUT_PATH.exists():
        # read existing schema and cast new batch to it (both should be FIXED_SCHEMA)
        existing = pq.read_table(OUT_PATH)
        # Safety: if existing isn’t fixed yet, cast our new table to existing’s schema
        if existing.schema != table.schema:
            table = table.cast(existing.schema)
        concat = pa.concat_tables([existing, table], promote=True)
        pq.write_table(concat, OUT_PATH)
    else:
        pq.write_table(table, OUT_PATH)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=CFG.get("external_paths", {}).get("reading_dir", ""),
                    help="Folder to scan (defaults to config.external_paths.reading_dir)")
    ap.add_argument("--max_docs", type=int, default=0, help="Limit number of files (0 = no limit)")
    args = ap.parse_args()

    base = Path(args.path).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Reading folder not found: {base}")

    exts = {".pdf", ".txt", ".epub"}
    files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if args.max_docs > 0:
        files = files[: args.max_docs]

    print(f"[info] scanning {len(files)} files under {base}")

    rows: list[dict] = []
    for fp in files:
        try:
            if fp.suffix.lower() == ".pdf":
                text = _parse_pdf(fp)
            elif fp.suffix.lower() == ".epub":
                text = _parse_epub(fp)
            else:
                text = _parse_txt(fp)
            if not text.strip():
                continue
            doc_id = _sha1_of_path(fp)
            rows.append({
                "doc_id": doc_id,
                "source": "reading",
                "path": str(fp),
                "text": text,
                # Optional extra columns; harmless if present:
                "title": fp.stem,
                "year": None,
                "authors": "",
                "venue": "",
            })
        except Exception as e:
            print(f"[warn] failed to parse {fp}: {e}")

        if len(rows) >= 500:
            _append_rows(rows)
            rows.clear()

    _append_rows(rows)
    print(f"[ok] ingested reading materials → {OUT_PARQUET}")

if __name__ == "__main__":
    main()
