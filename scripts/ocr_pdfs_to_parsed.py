# scripts/ocr_pdfs_to_parsed.py
from __future__ import annotations
import argparse, re, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        cp = subprocess.CompletedProcess(cmd, returncode=124, stdout=e.stdout or "", stderr="TIMEOUT")
        return cp

def pdftotext_text(pdf: Path) -> str:
    try:
        out = subprocess.check_output(["pdftotext", "-layout", "-q", str(pdf), "-"],
                                      stderr=subprocess.DEVNULL, timeout=90)
        return out.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def ocr_pdf(pdf: Path, dpi: int, lang: str, max_pages: int,
            timeout_render: int, timeout_ocr: int) -> str:
    # Render pages with pdftoppm (bounded), OCR each with tesseract (bounded)
    with tempfile.TemporaryDirectory(prefix="ocr_") as tmpd:
        prefix = Path(tmpd) / "page"
        f_l_args = []
        if max_pages and max_pages > 0:
            f_l_args = ["-f", "1", "-l", str(max_pages)]
        r = run(["pdftoppm", "-r", str(dpi), "-png", *f_l_args, str(pdf), str(prefix)],
                timeout=timeout_render)
        if r.returncode != 0:
            return ""

        def page_no(p: Path) -> int:
            m = re.search(r"-(\d+)\.png$", p.name)
            return int(m.group(1)) if m else 0

        pngs = sorted(Path(tmpd).glob("page-*.png"), key=page_no)
        texts = []
        for i, img in enumerate(pngs, 1):
            outbase = img.with_suffix("")  # -> outbase.txt
            rr = run(["tesseract", str(img), str(outbase), "-l", lang, "--oem", "1", "--psm", "1"],
                     timeout=timeout_ocr)
            if rr.returncode != 0:
                continue
            try:
                t = (outbase.with_suffix(".txt")).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                t = ""
            texts.append(f"\n\n=== PAGE {i} ===\n\n{t.strip()}")
        return "\n".join(texts).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", default="rag_store/parsed/parsed_docs.parquet")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--max-pages", type=int, default=0, help="0=all pages")
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="Skip docs already in output")
    ap.add_argument("--timeout-render", type=int, default=120, help="Seconds for pdftoppm")
    ap.add_argument("--timeout-ocr", type=int, default=45, help="Seconds per page for tesseract")
    ap.add_argument("--only-regex", default="", help="Only process PDFs whose path matches this regex")
    ap.add_argument("--skip-regex", default="", help="Skip PDFs whose path matches this regex")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    outp = Path(args.out)
    pdfs = sorted(in_dir.rglob("*.pdf"))
    if args.only_regex:
        rx = re.compile(args.only_regex, re.I)
        pdfs = [p for p in pdfs if rx.search(str(p))]
    if args.skip_regex:
        rx = re.compile(args.skip_regex, re.I)
        pdfs = [p for p in pdfs if not rx.search(str(p))]
    if args.limit and args.limit > 0:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        sys.exit(f"[error] no PDFs to process under {in_dir}")

    # Load existing doc_ids if --resume and output exists
    done_ids: set[str] = set()
    if args.resume and outp.exists():
        try:
            prev = pq.read_table(outp).to_pandas()
            done_ids = set(prev["doc_id"].astype(str))
            print(f"[resume] will skip {len(done_ids)} already-written docs")
        except Exception:
            pass

    schema = pa.schema([
        ("doc_id", pa.string()),
        ("source", pa.string()),
        ("path", pa.string()),
        ("text", pa.string()),
    ])
    outp.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(outp, schema, use_dictionary=False)

    written = 0
    empty_after_ocr = 0
    print(f"[info] OCR pipeline on {len(pdfs)} PDFs (dpi={args.dpi}, lang={args.lang}, max_pages={args.max_pages or 'all'})")
    try:
        for ix, pdf in enumerate(pdfs, 1):
            doc_id = pdf.stem
            if doc_id in done_ids:
                print(f"[{ix}/{len(pdfs)}] SKIP (resume) {pdf.name}")
                continue

            print(f"[{ix}/{len(pdfs)}] {pdf.name}")
            text = "" if args.force_ocr else pdftotext_text(pdf)
            if not text.strip():
                text = ocr_pdf(pdf, dpi=args.dpi, lang=args.lang,
                               max_pages=args.max_pages,
                               timeout_render=args.timeout_render,
                               timeout_ocr=args.timeout_ocr)
            if not text.strip():
                empty_after_ocr += 1

            row = pd.DataFrame([{
                "doc_id": doc_id,
                "source": "reading",
                "path": str(pdf),
                "text": text or ""
            }])
            writer.write_table(pa.Table.from_pandas(row))
            written += 1
    finally:
        writer.close()

    print(f"[ok] wrote {written} rows → {outp} | empty_after_ocr={empty_after_ocr}")

if __name__ == "__main__":
    main()
