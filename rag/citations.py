# rag/citations.py
from pathlib import Path
from typing import Dict, Tuple
from transformers import AutoTokenizer
import fitz  # PyMuPDF
import pandas as pd

AVG_TOKENS_PER_PAGE_BOOK = 450  # heuristic for pseudo pages in books

def _page_token_boundaries(pdf_path: Path, tokenizer):
    doc = fitz.open(pdf_path)
    page_texts, cumulative = [], [0]
    for page in doc:
        t = page.get_text("text")
        page_texts.append(t)
        toks = tokenizer.encode(t, add_special_tokens=False)
        cumulative.append(cumulative[-1] + len(toks))
    doc.close()
    return page_texts, cumulative

def _token_to_page(tok_index: int, cumulative: list) -> int:
    lo, hi = 0, len(cumulative) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cumulative[mid] <= tok_index < cumulative[mid + 1]:
            return mid
        if tok_index >= cumulative[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return max(0, min(lo, len(cumulative) - 2))

def _paragraph_index_in_text(text: str, span_text: str) -> int:
    paras = [p for p in text.split("\n\n") if p.strip()]
    if not paras:
        return 1
    key = span_text.strip().split(".")[0].lower()[:80]
    if key:
        for i, p in enumerate(paras, start=1):
            if key in p.lower():
                return i
    return 1

class CitationMapper:
    """
    Maps (source, path, tok_start..tok_end) to page + paragraph numbers.
    For PDFs: real page numbers. For books: paragraph in the whole doc,
    plus an optional pseudo page estimate.
    """
    def __init__(self, embed_model_name: str = "BAAI/bge-base-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self._pdf_cache = {}   # pdf_path -> (page_texts, cumulative_tokens)
        self._doc_texts = None # lazy-loaded dataframe of parsed docs

    def _ensure_doc_texts(self, root: Path):
        if self._doc_texts is None:
            parsed = root / "rag_store" / "parsed" / "parsed_docs.parquet"
            if parsed.exists():
                df = pd.read_parquet(parsed)
                # Keep only what's needed for fast lookup
                self._doc_texts = dict(zip(df["doc_id"].astype(str), df["text"].astype(str)))
            else:
                self._doc_texts = {}

    def cite(self, item: Dict) -> Dict:
        """
        item expects keys:
          source: 'jstor'|'book'|'s2orc'
          path: filesystem path or ""
          doc_id: str
          tok_start: int
          tok_end: int
          text: chunk text
        """
        src = item["source"]
        p = Path(item["path"]) if item["path"] else None

        # PDFs (JSTOR): map to real page + paragraph
        if src == "jstor" and p and p.suffix.lower() == ".pdf" and p.exists():
            if p not in self._pdf_cache:
                self._pdf_cache[p] = _page_token_boundaries(p, self.tokenizer)
            page_texts, cumulative = self._pdf_cache[p]
            page_num = _token_to_page(item["tok_start"], cumulative) + 1  # 1-based
            page_txt = page_texts[page_num - 1] if 0 < page_num <= len(page_texts) else ""
            para_num = _paragraph_index_in_text(page_txt, item["text"][:300])
            return {"page": page_num, "paragraph": para_num}

        # Books / S2ORC (plain text): compute paragraph number in the *whole doc*
        root = Path(__file__).resolve().parents[1]
        self._ensure_doc_texts(root)
        doc_text = self._doc_texts.get(str(item.get("doc_id")), "")

        if doc_text:
            # Count how many tokens occur before tok_start, then estimate paragraph index
            toks = self.tokenizer.encode(doc_text, add_special_tokens=False)
            # Map token index to character window for a cheap paragraph count
            # (approximate: decode up to tok_start and count double-newline blocks)
            cut = max(0, min(item["tok_start"], len(toks)))
            prefix_text = self.tokenizer.decode(toks[:cut])
            paras_before = [p for p in prefix_text.split("\n\n") if p.strip()]
            this_para = _paragraph_index_in_text(doc_text, item["text"][:300])  # refine within page
            paragraph_index = max(1, len(paras_before) + this_para)
        else:
            # Fallback to paragraph index within the chunk itself
            paras = [p for p in item["text"].split("\n\n") if p.strip()]
            paragraph_index = 1 if not paras else 1

        # Optional: pseudo page estimate for books (so it's never null)
        pseudo_page = 1 + (item["tok_start"] // AVG_TOKENS_PER_PAGE_BOOK)

        return {"page": pseudo_page, "paragraph": paragraph_index}
