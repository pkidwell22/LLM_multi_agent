# scripts/ingest_gutenberg_subjects.py
from __future__ import annotations

import argparse
import time
import re
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = ROOT / "rag_store" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)
OUT = PARSED_DIR / "parsed_docs.parquet"

# -----------------------------
# IO helpers
# -----------------------------
def _append_rows(rows: List[Dict]) -> None:
    """Append new rows to parsed_docs.parquet with stable schema."""
    if not rows:
        return

    # Force string type for 'year' in new rows
    for r in rows:
        r["year"] = str(r.get("year") or "")

    df_new = pd.DataFrame(rows)
    table_new = pa.Table.from_pandas(df_new)

    if OUT.exists():
        existing = pq.read_table(OUT)

        # 🔹 Coerce existing 'year' column to string if it's not already
        if existing.schema.field("year").type != pa.string():
            df_existing = existing.to_pandas()
            df_existing["year"] = df_existing["year"].astype(str)
            existing = pa.Table.from_pandas(df_existing)

        pq.write_table(pa.concat_tables([existing, table_new]), OUT)
    else:
        pq.write_table(table_new, OUT)


def _load_existing_ids() -> Set[str]:
    if not OUT.exists():
        return set()
    df = pd.read_parquet(OUT, columns=["doc_id"])
    return set(df["doc_id"].astype(str).tolist())

# -----------------------------
# Gutendex helpers
# -----------------------------
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "LLM-Multi-Agent/1.0 (+https://local)",
            "Accept": "application/json, text/plain, */*",
        }
    )
    return s

def _pick_plain_text_url(formats: Dict[str, str]) -> Optional[str]:
    """
    Prefer:
      1) text/plain; charset=utf-8 (not zipped)
      2) any text/plain (not zipped)
      3) text/html (not zipped) — we’ll strip tags
    """
    for mime, url in formats.items():
        if "text/plain" in mime and "utf-8" in mime.lower() and "zip" not in url.lower():
            return url
    for mime, url in formats.items():
        if "text/plain" in mime and "zip" not in url.lower():
            return url
    for mime, url in formats.items():
        if "text/html" in mime and "zip" not in url.lower():
            return url
    return None

TAG_RE = re.compile(r"<[^>]+>")

def _maybe_strip_html(raw: str) -> str:
    if "<" in raw and ">" in raw and ("</" in raw[:2000] or "<p" in raw[:2000]):
        return unescape(TAG_RE.sub(" ", raw))
    return raw

def _fetch_pages(
    base_url: str,
    english_only: bool,
    min_words: int,
    max_pages: int,
    max_items: int,
    sleep_s: float,
    existing_ids: Set[str],
    source_tag: str,
) -> int:
    """
    Shared paginator for both subject and keyword endpoints.
    base_url is a full Gutendex URL (already includes ?topic=... or ?search=...).
    """
    s = _session()
    url = base_url
    page_idx = 0
    total_added = 0
    batch: List[Dict] = []

    while url and (max_pages <= 0 or page_idx < max_pages) and (max_items <= 0 or total_added < max_items):
        try:
            r = s.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[warn] {source_tag}: request failed ({e}); retrying…")
            time.sleep(2.0)
            continue

        page_idx += 1
        books = data.get("results", [])
        print(f"[info] {source_tag}: page {page_idx} → {len(books)} results; kept so far={total_added}")

        for book in books:
            if max_items > 0 and total_added >= max_items:
                break

            if english_only:
                langs = book.get("languages") or []
                if "en" not in langs:
                    continue

            bid = str(book.get("id"))
            doc_id = f"gutenberg_{bid}"
            if doc_id in existing_ids:
                continue

            txt_url = _pick_plain_text_url(book.get("formats", {}))
            if not txt_url:
                continue

            try:
                raw = s.get(txt_url, timeout=50).text
            except Exception:
                continue

            raw = _maybe_strip_html(raw)
            wc = len(raw.split())
            if wc < min_words:
                title = (book.get("title") or "").strip()
                print(f"[skip] {source_tag}: {title[:70]!r} ({wc} words) < min_words={min_words}")
                continue

            title = (book.get("title") or "").strip()
            authors = ", ".join(a.get("name", "") for a in (book.get("authors") or []))
            year = (book.get("authors") or [{}])[0].get("birth_year")

            batch.append(
                {
                    "doc_id": doc_id,
                    "source": "gutenberg",
                    "path": txt_url,
                    "text": raw,
                    "title": title,
                    "year": str(year or ""),
                    "authors": authors,
                    "venue": "Project Gutenberg",
                }
            )
            existing_ids.add(doc_id)
            total_added += 1

            if len(batch) >= 100:
                _append_rows(batch)
                print(f"[info] {source_tag}: appended → total {total_added}")
                batch.clear()

        url = data.get("next")
        if not url:
            print(f"[info] {source_tag}: no more pages.")
        if url and sleep_s > 0:
            time.sleep(sleep_s)

    if batch:
        _append_rows(batch)
        print(f"[info] {source_tag}: final append {len(batch)}")

    print(f"[ok] {source_tag}: added {total_added} docs → {OUT}")
    return total_added

# -----------------------------
# Subject fetch
# -----------------------------
def fetch_by_subjects(
    subjects: List[str],
    min_words: int,
    max_pages: int,
    max_per_subject: int,
    sleep_s: float,
    english_only: bool,
    existing_ids: Set[str],
) -> int:
    total = 0
    for sub in subjects:
        base = f"https://gutendex.com/books?topic={requests.utils.quote(sub)}"
        total += _fetch_pages(
            base_url=base,
            english_only=english_only,
            min_words=min_words,
            max_pages=max_pages,
            max_items=max_per_subject,
            sleep_s=sleep_s,
            existing_ids=existing_ids,
            source_tag=f"subject:{sub}",
        )
    return total

# -----------------------------
# Keyword search
# -----------------------------
def fetch_by_keywords(
    keywords: List[str],
    min_words: int,
    max_pages: int,
    max_per_keyword: int,
    sleep_s: float,
    english_only: bool,
    existing_ids: Set[str],
) -> int:
    total = 0
    for kw in keywords:
        base = f"https://gutendex.com/books?search={requests.utils.quote(kw)}"
        total += _fetch_pages(
            base_url=base,
            english_only=english_only,
            min_words=min_words,
            max_pages=max_pages,
            max_items=max_per_keyword,
            sleep_s=sleep_s,
            existing_ids=existing_ids,
            source_tag=f"keyword:{kw}",
        )
    return total

# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default="philosophy,science",
                    help='Comma-separated subjects, e.g. "philosophy,science"')
    ap.add_argument("--keywords", type=str, default="",
                    help='Comma-separated keywords, e.g. "metaphysics,epistemology,stoicism,physics"')
    ap.add_argument("--min_words", type=int, default=4500)
    ap.add_argument("--max_pages", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max_per_subject", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max_per_keyword", type=int, default=0, help="0 = no limit")
    ap.add_argument("--sleep", type=float, default=0.6, help="seconds between pages")
    ap.add_argument("--english_only", action="store_true", default=True)  # default True
    ap.add_argument("--no_english_only", dest="english_only", action="store_false")
    ap.add_argument("--skip_subjects", action="store_true")
    ap.add_argument("--skip_keywords", action="store_true")
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    print(
        f"[info] starting Gutenberg crawl "
        f"(subjects={subjects}, keywords={keywords}, min_words={args.min_words}, english_only={args.english_only})"
    )

    existing_ids = _load_existing_ids()

    total_added = 0
    if not args.skip_subjects and subjects:
        total_added += fetch_by_subjects(
            subjects=subjects,
            min_words=args.min_words,
            max_pages=args.max_pages,
            max_per_subject=args.max_per_subject,
            sleep_s=args.sleep,
            english_only=args.english_only,
            existing_ids=existing_ids,
        )

    if not args.skip_keywords and keywords:
        total_added += fetch_by_keywords(
            keywords=keywords,
            min_words=args.min_words,
            max_pages=args.max_pages,
            max_per_keyword=args.max_per_keyword,
            sleep_s=args.sleep,
            english_only=args.english_only,
            existing_ids=existing_ids,
        )

    print(f"[ok] Gutenberg ingestion complete. Added {total_added} new docs.")

if __name__ == "__main__":
    main()
