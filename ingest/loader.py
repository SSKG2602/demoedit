#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Ingestion Loader
Purpose:
  • Read plain-text sources (.txt, .md; optional .pdf if pypdf is available)
  • Normalize & chunk into passage windows with small overlaps
  • Emit disk artifacts for downstream indexing:
      - data/docs.csv      (one row per source file)
      - data/chunks.csv    (one row per chunk)
Design:
  • Zero deviations from PoC/RG docs: store minimal time metadata now;
    PVDB/BiTX etc. live outside Lite. We keep CSVs lean for FAISS build.
  • Low-RAM friendly (no heavy NLP libs; pure-Python + pandas).
  • Deterministic, debug-friendly; rich logging.

CLI:
  python -m ingest.loader --path ./ingest/demo_dataset \
                          --chunk_size 400 --overlap 40 \
                          --tz Asia/Kolkata
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import pandas as pd

# Optional PDF support (kept optional to avoid hard deps in the env)
try:
    from pypdf import PdfReader  # type: ignore
    _PDF_OK = True
except Exception:
    _PDF_OK = False

# ---------------------------
# Simple, local configuration
# (kept here to avoid cross-module import order issues; mirror these in app/settings.py if desired)
# ---------------------------
# ~words per chunk (Lite: 200–400 ok; RG uses higher)
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 40   # ~words overlap
# PoC default; alias CT→America/Chicago handled in temporal module
DEFAULT_TZ = "Asia/Kolkata"
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
_WHITESPACE = re.compile(r"[ \t]+")
_NEWLINES = re.compile(r"\n{3,}")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[loader {ts}] {msg}", flush=True)


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return "text/plain"
    if ext == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_pdf_text(path: Path) -> str:
    if not _PDF_OK:
        raise RuntimeError(
            "PDF support not available (pypdf not installed). "
            "Install with `pip install pypdf` or pre-convert PDFs to .txt"
        )
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def normalize_text(s: str) -> str:
    # collapse weird whitespace; keep single blank lines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WHITESPACE.sub(" ", s)
    s = _NEWLINES.sub("\n\n", s)
    return s.strip()


def word_tokenize(text: str) -> List[str]:
    # lightweight, transparent tokenization (space-punct split)
    # good enough for chunk length control without extra deps
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def detokenize(words: List[str]) -> str:
    # simple join with space; punctuation will have spaces—acceptable for retrieval
    return " ".join(words).strip()


def chunk_words(words: List[str], chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start_idx, end_idx, text)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size-1]")
    chunks = []
    n = len(words)
    i = 0
    while i < n:
        j = min(i + chunk_size, n)
        window = detokenize(words[i:j])
        chunks.append((i, j, window))
        if j == n:
            break
        i = j - overlap  # step with overlap
    return chunks


def file_observed_at(path: Path) -> str:
    # mtime as observed_at; RFC3339
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.isoformat()

# ---------------------------
# Core ingestion
# ---------------------------


def ingest_path(
    root: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    tz_str: str = DEFAULT_TZ,
    default_published: Optional[str] = None,
    default_valid_from: Optional[str] = None,
    default_valid_to: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan a folder, load supported files, chunk them, and return (docs_df, chunks_df).
    """
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    doc_rows: List[Dict] = []
    chunk_rows: List[Dict] = []

    files = sorted(
        [p for p in root.rglob("*") if p.is_file()
         and p.suffix.lower() in (".txt", ".md", ".pdf")]
    )
    if not files:
        _log(f"No .txt/.md/.pdf files under: {root}")
        return pd.DataFrame(), pd.DataFrame()

    _log(f"Found {len(files)} candidate files under {root}")

    for fpath in files:
        try:
            mime = guess_mime(fpath)
            if mime == "text/plain":
                raw = read_text_file(fpath)
            elif mime == "application/pdf":
                raw = read_pdf_text(fpath)
            else:
                _log(f"Skipping unsupported file: {fpath.name}")
                continue

            text = normalize_text(raw)
            if not text:
                _log(f"Empty after normalize → skip: {fpath.name}")
                continue

            words = word_tokenize(text)
            chunks = chunk_words(words, chunk_size, overlap)

            # Document identity
            doc_id = sha1_hex(str(fpath.resolve()))
            observed_at = file_observed_at(fpath)

            # Time fields are lightweight placeholders in Lite (PoC supports simple fields)
            published_at = default_published  # can be None
            valid_from = default_valid_from
            valid_to = default_valid_to

            # Doc row
            doc_rows.append(
                dict(
                    doc_id=doc_id,
                    source_path=str(fpath),
                    mime=mime,
                    bytes=fpath.stat().st_size,
                    observed_at=observed_at,
                    published_at=published_at,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    tz=tz_str,
                )
            )

            # Chunk rows
            for idx, (w0, w1, chunk_text) in enumerate(chunks):
                chunk_id = sha1_hex(f"{doc_id}:{idx}:{w0}-{w1}")
                chunk_rows.append(
                    dict(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        chunk_index=idx,
                        w_start=w0,
                        w_end=w1,
                        text=chunk_text,
                        observed_at=observed_at,
                        published_at=published_at,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        tz=tz_str,
                        source_path=str(fpath),
                    )
                )

            _log(f"Ingested {fpath.name}: {len(chunks)} chunks")

        except KeyboardInterrupt:
            _log("Interrupted by user.")
            raise
        except Exception as e:
            _log(f"ERROR {fpath.name}: {e}")

    docs_df = pd.DataFrame(doc_rows)
    chunks_df = pd.DataFrame(chunk_rows)
    return docs_df, chunks_df


def save_outputs(docs_df: pd.DataFrame, chunks_df: pd.DataFrame) -> None:
    if docs_df.empty and chunks_df.empty:
        _log("Nothing to save.")
        return
    docs_csv = DATA_DIR / "docs.csv"
    chunks_csv = DATA_DIR / "chunks.csv"
    docs_df.to_csv(docs_csv, index=False)
    chunks_df.to_csv(chunks_csv, index=False)
    _log(f"Wrote: {docs_csv}  ({len(docs_df)} rows)")
    _log(f"Wrote: {chunks_csv} ({len(chunks_df)} rows)")

# ---------------------------
# CLI
# ---------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ingest.loader",
        description="ChronoRAG-Lite ingestion loader (text/md/pdf -> normalized chunks CSV)",
    )
    p.add_argument("--path", required=True,
                   help="Folder containing .txt/.md/.pdf files")
    p.add_argument("--chunk_size", type=int,
                   default=DEFAULT_CHUNK_SIZE, help="~words per chunk")
    p.add_argument("--overlap", type=int,
                   default=DEFAULT_CHUNK_OVERLAP, help="~words overlap")
    p.add_argument("--tz", type=str, default=DEFAULT_TZ,
                   help="Timezone tag stored with rows")
    p.add_argument("--published", type=str, default=None,
                   help="Default published_at (YYYY-MM-DD)")
    p.add_argument("--valid_from", type=str, default=None,
                   help="Default valid_from (YYYY-MM-DD)")
    p.add_argument("--valid_to", type=str, default=None,
                   help="Default valid_to (YYYY-MM-DD)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.path).expanduser().resolve()

    _log(
        f"Start ingest: root={root} chunk_size={args.chunk_size} overlap={args.overlap} tz={args.tz}")
    t0 = time.time()
    docs, chunks = ingest_path(
        root=root,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        tz_str=args.tz,
        default_published=args.published,
        default_valid_from=args.valid_from,
        default_valid_to=args.valid_to,
    )
    save_outputs(docs, chunks)
    _log(f"Done in {time.time()-t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
