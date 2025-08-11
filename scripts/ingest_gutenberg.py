# scripts/ingest_gutenberg.py
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ebooklib import epub

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "rag_store"
RAW = DATA / "raw" / "books" / "gutenberg"
PARSED = DATA / "parsed"
PARSED.mkdir(parents=True, exist_ok=True)
OUT = PARSED / "parsed_docs.parquet"

GUTENDEX = "https://gutendex.com/books"

# Minimal starter canon (still works with --mode curated)
CURATED = [
    {"author": "Homer", "title": "Iliad"},
    {"author": "Homer", "title": "Odyssey"},
    {"author": "Virgil", "title": "Aeneid"},
    {"author": "Dante Alighieri", "title": "Divine Comedy"},
    {"author": "Plato", "title": "Republic"},
    {"author": "Aristotle", "title": "Poetics"},
    {"author": "Miguel de Cervantes", "title": "Don Quixote"},
    {"author": "William Shakespeare", "title": "Hamlet"},
    {"author": "John Milton", "title": "Paradise Lost"},
    {"author": "Jane Austen", "title": "Pride and Prejudice"},
    {"author": "Mary Shelley", "title": "Frankenstein"},
    {"author": "Victor Hugo", "title": "Les Misérables"},
    {"author": "Fyodor Dostoevsky", "title": "Crime and Punishment"},
    {"author": "Leo Tolstoy", "title": "War and Peace"},
    {"author": "Charles Dickens", "title": "Great Expectations"},
]

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def _short_fname(gid: int, title: str, is_txt: bool) -> str:
    # Windows path guard: keep basename comfortably short
    base = f"{gid}_{_slug(title)}"
    base = base[:120]  # prevent >260 path issues
    ext = "txt" if is_txt else "epub"
    if not base:
        base = str(gid)
    return f"{base}.{ext}"

def _strip_gutenberg_boiler(text: str) -> str:
    start_pat = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG", re.I)
    end_pat = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG", re.I)
    start = start_pat.search(text)
    end = end_pat.search(text)
    if start:
        text = text[start.end():]
    if end:
        text = text[:end.start()]
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _pick_best_format(formats: Dict[str, str]) -> Optional[Dict[str, str]]:
    # Prefer UTF‑8 txt, then txt, then epub (no images)
    prefer = [
        "text/plain; charset=utf-8",
        "text/plain",
        "application/epub+zip",
    ]
    for key in prefer:
        for k, url in formats.items():
            if k.lower().startswith(key) and url and url.lower().startswith(("http://", "https://")):
                return {"mime": k, "url": url}
    return None

def _download_txt(url: str) -> Optional[str]:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    return r.content.decode("utf-8", errors="ignore")

def _download_epub(url: str, save_path: Path) -> Optional[str]:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    save_path.write_bytes(r.content)
    try:
        book = epub.read_epub(str(save_path))
        parts: List[str] = []
        for item in book.get_items():
            if getattr(item, "get_type", lambda: None)() == 9:  # DOCUMENT
                try:
                    parts.append(item.get_body_content().decode("utf-8", errors="ignore"))
                except Exception:
                    pass
        html = " ".join(parts)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return None

def _append_rows(rows: List[Dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    if OUT.exists():
        existing = pq.read_table(OUT)
        pq.write_table(pa.concat_tables([existing, table]), OUT)
    else:
        pq.write_table(table, OUT)

def _existing_doc_ids() -> Set[str]:
    if not OUT.exists():
        return set()
    try:
        df = pq.read_table(OUT, columns=["doc_id"]).to_pandas()
        return set(df["doc_id"].astype(str).tolist())
    except Exception:
        return set()

def _search_gutendex(params: Dict) -> Dict:
    r = requests.get(GUTENDEX, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def _pick_candidate(results: Dict, want_author: str, want_title: str) -> Optional[Dict]:
    items = results.get("results", [])
    want_author_l = want_author.lower()
    want_title_l = want_title.lower()
    def score(item):
        lang_ok = "en" in item.get("languages", [])
        auth_match = any(want_author_l in a.get("name", "").lower() for a in item.get("authors", []))
        title_match = want_title_l in (item.get("title") or "").lower()
        return (1 if lang_ok else 0, 1 if auth_match else 0, 1 if title_match else 0, item.get("download_count", 0))
    if not items:
        return None
    return sorted(items, key=score, reverse=True)[0]

def _save_one(sess: requests.Session, item: Dict, sleep: float, min_words: int, existing: Set[str]) -> Optional[Dict]:
    gid = int(item["id"])
    doc_id = f"gutenberg_{gid}"
    if doc_id in existing:
        return None

    formats = item.get("formats", {}) or {}
    pick = _pick_best_format(formats)
    if not pick:
        return None

    is_txt = pick["mime"].startswith("text/")
    fname = _short_fname(gid, item.get("title") or str(gid), is_txt=is_txt)
    RAW.mkdir(parents=True, exist_ok=True)
    tmp_path = RAW / fname

    print(f"[info] downloading Gutenberg #{gid} → {tmp_path.name} …")
    if is_txt:
        text = _download_txt(pick["url"])
        if not text:
            return None
        # write only after we confirm; avoids partial files
        tmp_path.write_text(text, encoding="utf-8")
    else:
        text = _download_epub(pick["url"], tmp_path)
        if not text:
            # bad epub; cleanup
            tmp_path.unlink(missing_ok=True)
            return None

    clean = _strip_gutenberg_boiler(text)
    if min_words and len(clean.split()) < min_words:
        # too short; discard file if we created it
        tmp_path.unlink(missing_ok=True)
        return None

    row = {
        "doc_id": doc_id,
        "source": "book",
        "path": str(tmp_path),
        "text": clean,
        "title": item.get("title", ""),
        "year": item.get("copyright_year") or None,
        "authors": ", ".join(a.get("name", "") for a in item.get("authors", [])),
        "venue": "Project Gutenberg",
    }
    time.sleep(max(0.0, sleep))
    return row

# ---------------- MODES ----------------

def run_curated(limit: Optional[int] = None, sleep: float = 0.6, min_words: int = 0):
    rows: List[Dict] = []
    todo = CURATED[: limit or len(CURATED)]
    existing = _existing_doc_ids()
    with requests.Session() as sess:
        for rec in todo:
            q = f'{rec["author"]} {rec["title"]}'
            try:
                data = _search_gutendex({"search": q})
            except Exception as e:
                print(f"[warn] search failed for {q}: {e}")
                continue
            item = _pick_candidate(data, rec["author"], rec["title"])
            if not item:
                print(f"[warn] no match for {q}")
                continue
            got = _save_one(sess, item, sleep, min_words, existing)
            if got:
                rows.append(got)
                existing.add(got["doc_id"])
    _append_rows(rows)
    print(f"[ok] added {len(rows)} curated books → {OUT}")

def run_query(q: str, max_books: int = 10, sleep: float = 0.5, min_words: int = 0):
    data = _search_gutendex({"search": q})
    items = data.get("results", [])[:max_books]
    rows: List[Dict] = []
    existing = _existing_doc_ids()
    with requests.Session() as sess:
        for it in items:
            got = _save_one(sess, it, sleep, min_words, existing)
            if got:
                rows.append(got)
                existing.add(got["doc_id"])
    _append_rows(rows)
    print(f"[ok] added {len(rows)} books from query → {OUT}")

def run_top(n: int, max_pages: int = 200, sleep: float = 0.6, min_words: int = 0):
    """
    Crawl Gutendex by popularity, English only, filter short works.
    """
    print(f"[info] gathering candidates from Gutendex pages (max_pages={max_pages}) …\n")
    rows: List[Dict] = []
    existing = _existing_doc_ids()
    count = 0
    page = 1
    with requests.Session() as sess:
        while count < n and page <= max_pages:
            try:
                data = _search_gutendex({"page": page})
            except Exception as e:
                print(f"[warn] page {page} failed: {e}")
                break
            items = data.get("results", [])
            if not items:
                break
            # sort page by download_count desc to prioritize
            items = sorted(items, key=lambda x: x.get("download_count", 0), reverse=True)
            for it in items:
                if count >= n:
                    break
                if "en" not in it.get("languages", []):
                    continue
                got = _save_one(sess, it, sleep, min_words, existing)
                if got:
                    rows.append(got)
                    existing.add(got["doc_id"])
                    count += 1
            page += 1
    _append_rows(rows)
    print(f"[ok] added {len(rows)} top‑mode books → {OUT}")

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["curated", "query", "top"], required=True)
    # curated/query
    ap.add_argument("--limit", type=int, default=None, help="limit curated list size")
    ap.add_argument("--q", type=str, default=None, help="search query for --mode query")
    ap.add_argument("--max_books", type=int, default=10, help="max books to fetch for --mode query")
    # top
    ap.add_argument("--n", type=int, default=200, help="how many books to fetch in --mode top")
    ap.add_argument("--max_pages", type=int, default=200, help="gutendex page limit for --mode top")
    # generic filters
    ap.add_argument("--min_words", type=int, default=0, help="drop works shorter than this many words")
    ap.add_argument("--sleep", type=float, default=0.6, help="delay between downloads (seconds)")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    if args.mode == "curated":
        run_curated(limit=args.limit, sleep=args.sleep, min_words=args.min_words)
    elif args.mode == "query":
        if not args.q:
            raise SystemExit("--q is required for --mode query")
        run_query(args.q, max_books=args.max_books, sleep=args.sleep, min_words=args.min_words)
    else:
        run_top(n=args.n, max_pages=args.max_pages, sleep=args.sleep, min_words=args.min_words)

if __name__ == "__main__":
    main()
