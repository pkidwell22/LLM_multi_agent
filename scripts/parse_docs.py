# scripts/parse_docs.py
import re
import subprocess
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fitz
from ebooklib import epub
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT/"config/settings.yaml").read_text())
DATA = ROOT/"rag_store"
RAW = DATA/"raw"
PARSED = DATA/"parsed"
PARSED.mkdir(exist_ok=True, parents=True)

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
    if needs_ocr and allow_ocr and CFG["ocr"]["enabled"]:
        return _ocr_pdf(pdf_path)
    return "\n".join(pages)

def _parse_epub(epub_path: Path) -> str:
    book = epub.read_epub(str(epub_path))
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

def main():
    rows = []
    for p in sorted((RAW/"books").glob("**/*")):
        if p.suffix.lower() not in [".txt", ".epub", ".pdf"]:
            continue
        if p.suffix.lower() == ".txt":
            text = _parse_txt(p)
        elif p.suffix.lower() == ".epub":
            text = _parse_epub(p)
        else:
            text = _parse_pdf(p, allow_ocr=True)
        rows.append({"doc_id": p.stem, "source": "book", "path": str(p), "text": text})

    for p in sorted((RAW/"jstor").glob("*.pdf")):
        text = _parse_pdf(p, allow_ocr=True)
        rows.append({"doc_id": p.stem, "source": "jstor", "path": str(p), "text": text})

    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), PARSED/"parsed_docs.parquet")
    print(f"[ok] parsed: {len(df)} documents → {PARSED/'parsed_docs.parquet'}")

if __name__ == "__main__":
    main()
