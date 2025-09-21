#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Unified Runner
Purpose:
  • Organize common ops in one script:
      - ingest   : read ./ingest/<folder> → write ./data/*.csv
      - index    : build FAISS from ./data/chunks.csv → ./indexes/faiss.index
      - ask      : interactive Q&A (direct orchestrator, no server)
      - serve    : start FastAPI (uvicorn) to use HTTP routes
      - quick    : ingest + index + ask (fast path)
  • Zero deviation from architecture — this is a thin orchestration layer.

Usage (examples):
  python run.py ingest --path ./ingest/demo_dataset
  python run.py index --model BAAI/bge-small-en-v1.5 --prefer_fastembed
  python run.py ask
  python run.py serve --port 8000
  python run.py quick --path ./ingest/demo_dataset

Notes:
  • All defaults match your existing modules.
  • Set env via .env or pass flags. See app/settings.py/.env.example.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# --- Local imports from your codebase (no external deps here) ---
# Ingest
from ingest.loader import ingest_path, save_outputs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TZ
# Index
from ingest.build_faiss import build_index as build_faiss_index
# Orchestration (retrieval + answer)
from app.orchestrator import (
    QueryRequest,
    retrieve_only as orch_retrieve,
    answer as orch_answer,
)
# Settings (paths/models)
from app.settings import settings


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[run {ts}] {msg}", flush=True)


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


# ------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest a folder of text/md/pdf into ./data/docs.csv + ./data/chunks.csv."""
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        _log(f"ERROR: path not found: {path}")
        return 2

    _log(
        f"Ingesting: {path} (chunk_size={args.chunk_size}, overlap={args.overlap}, tz={args.tz})")
    docs_df, chunks_df = ingest_path(
        root=path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        tz_str=args.tz,
        default_published=args.published,
        default_valid_from=args.valid_from,
        default_valid_to=args.valid_to,
    )
    save_outputs(docs_df, chunks_df)
    _log("Ingest complete.")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Build a FAISS index from ./data/chunks.csv and persist artifacts."""
    chunks_csv = Path(args.chunks or settings.chunks_csv)
    index_out = Path(args.index_out or settings.faiss_index)
    model = args.model or settings.embed_model
    prefer_fastembed = bool(args.prefer_fastembed)

    if not _exists(chunks_csv):
        _log(
            f"ERROR: missing chunks CSV: {chunks_csv}. Run `python run.py ingest --path <folder>` first.")
        return 2

    _log(f"Building FAISS index → {index_out}")
    build_faiss_index(
        chunks_csv=chunks_csv,
        model_name=model,
        dim=args.dim,
        batch=args.batch,
        prefer_fastembed=prefer_fastembed,
        index_out=index_out,
    )
    _log("Index build complete.")
    return 0


def _ensure_ready_for_query() -> Optional[str]:
    """Return a message if something is missing; otherwise None."""
    need = []
    if not _exists(settings.chunks_csv):
        need.append(str(settings.chunks_csv))
    if not _exists(settings.faiss_index):
        need.append(str(settings.faiss_index))
    if not _exists(settings.ids_npy):
        need.append(str(settings.ids_npy))
    if not _exists(settings.meta_parquet):
        need.append(str(settings.meta_parquet))
    if need:
        return "Missing artifacts:\n  - " + "\n  - ".join(need)
    return None


def cmd_ask(args: argparse.Namespace) -> int:
    """
    Interactive Q&A loop using the orchestrator directly (no server).
    - Uses current settings (models, weights, paths).
    - Answers via local HF LLM if available, otherwise deterministic fallback.
    """
    missing = _ensure_ready_for_query()
    if missing:
        _log("ERROR: " + missing)
        _log("Run ingest + index first. Example:\n"
             "  python run.py ingest --path ./ingest/demo_dataset\n"
             "  python run.py index --prefer_fastembed")
        return 2

    _log("Interactive Q&A (type 'exit' or 'quit' to stop).")
    _log(f"Model: embed={settings.embed_model}  llm={settings.llm_model}")
    try:
        while True:
            q = input("\nQ> ").strip()
            if q.lower() in {"exit", "quit"}:
                _log("Bye.")
                break
            if not q:
                continue

            # Retrieval first (for inspection)
            req = QueryRequest(query=q, k=args.k, time_mode=args.time_mode,
                               prefer_fastembed=settings.prefer_fastembed)
            t0 = time.time()
            if args.retrieval_only:
                resp = orch_retrieve(req)
                dt = (time.time() - t0) * 1000
                print(f"\n[Retrieval-only | {dt:.0f} ms]")
                print(f"MODE={resp.time_filters.mode}  WINDOW=[{resp.time_filters.start} .. {resp.time_filters.end})  "
                      f"CONF={resp.time_filters.confidence:.2f}")
                for i, c in enumerate(resp.candidates, 1):
                    print(
                        f"{i:02d}. total={c.score_total:.3f}  sim={c.score_sim:.3f}  time={c.score_time:.3f}  age={c.score_age:.3f}")
                    print(f"    {c.text[:160].replace(chr(10),' ')}")
                    if c.source_path:
                        print(f"    src={c.source_path}")
                continue

            # Full answer
            resp = orch_answer(req)
            dt = (time.time() - t0) * 1000
            print(
                f"\n[Answer | {dt:.0f} ms] MODE={resp.time_filters.mode}  WINDOW=[{resp.time_filters.start} .. {resp.time_filters.end})")
            print("\n--- ANSWER ---\n")
            print(resp.answer)

            # Show top-3 evidence receipts
            print("\n--- TOP EVIDENCE ---\n")
            for i, c in enumerate(resp.candidates[:3], 1):
                line = c.text[:200].replace("\n", " ")
                src = c.source_path or c.doc_id
                print(
                    f"{i:02d}. {line}  (src={src})  [sim={c.score_sim:.3f} time={c.score_time:.3f} age={c.score_age:.3f}]")

    except KeyboardInterrupt:
        _log("Interrupted.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """
    Start FastAPI via uvicorn, using app.api:app (your existing API).
    Note: this blocks the terminal; run in a separate shell or use a process manager if needed.
    """
    # Import here to avoid uvicorn import if not used
    try:
        import uvicorn  # type: ignore
    except Exception as e:
        _log(
            f"ERROR: uvicorn not found in env ({e}). Install via conda/pip and retry.")
        return 2

    host = args.host or str(settings.api_host)
    port = int(args.port) if args.port else int(settings.api_port)

    _log(f"Starting API at http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("app.api:app", host=host, port=port, reload=args.reload)
    return 0


def cmd_quick(args: argparse.Namespace) -> int:
    """
    Quick path for a new corpus:
      1) ingest --path <folder>
      2) index  --prefer_fastembed
      3) ask    (interactive loop)
    """
    # 1) Ingest
    rc = cmd_ingest(args)
    if rc != 0:
        return rc
    # 2) Index
    idx_args = argparse.Namespace(
        chunks=None, index_out=None, model=args.model, dim=None,
        batch=args.batch, prefer_fastembed=args.prefer_fastembed
    )
    rc = cmd_index(idx_args)
    if rc != 0:
        return rc
    # 3) Ask
    ask_args = argparse.Namespace(
        k=args.k, time_mode=args.time_mode, retrieval_only=False)
    return cmd_ask(ask_args)


# ------------------------------------------------------------------------------
# CLI Parser
# ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="ChronoRAG-Lite unified runner (ingest, index, ask, serve, quick).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ingest
    sp = sub.add_parser("indigest", help=argparse.SUPPRESS)  # alias typo guard
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser(
        "ingest", help="Ingest a folder of .txt/.md/.pdf into ./data/*.csv")
    sp.add_argument("--path", required=True,
                    help="Folder containing corpus files")
    sp.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    sp.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    sp.add_argument("--tz", type=str, default=DEFAULT_TZ)
    sp.add_argument("--published", type=str, default=None)
    sp.add_argument("--valid_from", type=str, default=None)
    sp.add_argument("--valid_to", type=str, default=None)
    sp.set_defaults(func=cmd_ingest)

    # index
    sp = sub.add_parser(
        "index", help="Build FAISS index from ./data/chunks.csv")
    sp.add_argument("--chunks", type=str, default=None,
                    help="Path to chunks.csv (default from settings)")
    sp.add_argument("--model", type=str, default=None,
                    help="Embedding model (default from settings)")
    sp.add_argument("--dim", type=int, default=None)
    sp.add_argument("--batch", type=int, default=64)
    sp.add_argument("--prefer_fastembed", action="store_true")
    sp.add_argument("--index_out", type=str, default=None,
                    help="Output index path (default from settings)")
    sp.set_defaults(func=cmd_index)

    # ask
    sp = sub.add_parser("ask", help="Interactive Q&A loop (no server)")
    sp.add_argument("--k", type=int, default=6)
    sp.add_argument("--time_mode", type=str, default=None,
                    choices=[None, "INTELLIGENT", "HARD"])
    sp.add_argument("--retrieval_only", action="store_true",
                    help="Skip LLM; show retrieval results only")
    sp.set_defaults(func=cmd_ask)

    # serve
    sp = sub.add_parser(
        "serve", help="Start FastAPI (uvicorn) at the configured host/port")
    sp.add_argument("--host", type=str, default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)

    # quick
    sp = sub.add_parser(
        "quick", help="Ingest + Index + Ask (fast path for a new corpus)")
    sp.add_argument("--path", required=True,
                    help="Folder with corpus files (e.g., ./ingest/demo_dataset)")
    sp.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    sp.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    sp.add_argument("--tz", type=str, default=DEFAULT_TZ)
    sp.add_argument("--published", type=str, default=None)
    sp.add_argument("--valid_from", type=str, default=None)
    sp.add_argument("--valid_to", type=str, default=None)
    sp.add_argument("--model", type=str, default=None,
                    help="Embedding model for indexing")
    sp.add_argument("--batch", type=int, default=64)
    sp.add_argument("--prefer_fastembed", action="store_true")
    sp.add_argument("--k", type=int, default=6)
    sp.add_argument("--time_mode", type=str, default=None,
                    choices=[None, "INTELLIGENT", "HARD"])
    sp.set_defaults(func=cmd_quick)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
