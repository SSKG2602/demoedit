#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — FAISS Index Builder
Purpose:
  • Load data/chunks.csv (produced by ingest.loader)
  • Compute embeddings for each chunk text
  • Build a cosine FAISS index (IndexFlatIP on unit vectors)
  • Persist index and metadata to disk

Design:
  • Low-RAM friendly: stream/batch embeds; no giant in-memory copies
  • Debuggable: saves embeddings.npy + chunk_ids.npy for inspection
  • Model provider:
      - Prefer fastembed (CPU, tiny, quick)
      - Fallback to sentence-transformers

CLI:
  python -m ingest.build_faiss --model BAAI/bge-small-en-v1.5 --batch 64 --dim 384
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import faiss  # conda: faiss-cpu

DATA_DIR = Path("data")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Logging
# ---------------------------


def _log(msg: str) -> None:
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[faiss {ts}] {msg}", flush=True)

# ---------------------------
# Embedding Providers
# ---------------------------


class Embedder:
    """Abstract embedder API."""

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class FastEmbedder(Embedder):
    """FastEmbed backend (if installed)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # type: ignore
        self.model = TextEmbedding(model_name=model_name)
        # fastembed exposes dim per model; fallback to probing one sample if needed
        try:
            self._dim = self.model.get_output_dim()  # type: ignore
        except Exception:
            vec = list(self.model.embed(["probe"]))[0]
            self._dim = len(vec)

    @property
    def dim(self) -> int:
        return int(self._dim)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # fastembed supports streaming
        embs = list(self.model.embed(texts))
        return np.asarray(embs, dtype=np.float32)


class STEmbedder(Embedder):
    """Sentence-Transformers fallback."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name, device=device or "cpu")
        # try to infer dimension
        try:
            self._dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            self._dim = len(self.model.encode(
                ["probe"], normalize_embeddings=False)[0])

    @property
    def dim(self) -> int:
        return int(self._dim)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return vecs.astype(np.float32, copy=False)


def resolve_embedder(model_name: str, prefer_fastembed: bool = True) -> Embedder:
    if prefer_fastembed:
        try:
            _log(f"Trying fastembed: {model_name}")
            return FastEmbedder(model_name)
        except Exception as e:
            _log(f"fastembed not available or failed: {e}")
    _log(f"Using sentence-transformers: {model_name}")
    return STEmbedder(model_name)

# ---------------------------
# Helpers
# ---------------------------


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def chunk_iter(df: pd.DataFrame, batch: int) -> Iterable[pd.DataFrame]:
    n = len(df)
    for i in range(0, n, batch):
        yield df.iloc[i: i + batch]

# ---------------------------
# Core build
# ---------------------------


def build_index(
    chunks_csv: Path,
    model_name: str,
    dim: Optional[int],
    batch: int,
    prefer_fastembed: bool,
    index_out: Path,
) -> None:
    if not chunks_csv.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {chunks_csv} (run ingest.loader first)")

    _log(f"Loading chunks: {chunks_csv}")
    df = pd.read_csv(chunks_csv)
    if df.empty:
        raise RuntimeError("chunks.csv is empty")

    # Select mandatory fields
    required_cols = ["chunk_id", "text",
                     "doc_id", "observed_at", "source_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"chunks.csv missing required columns: {missing}")

    # Prepare embedder
    emb = resolve_embedder(model_name, prefer_fastembed=prefer_fastembed)
    if dim is not None and int(dim) != emb.dim:
        _log(
            f"WARNING: provided --dim {dim} != model dim {emb.dim}. Using model dim.")
    dim = emb.dim

    # Create FAISS index (cosine via inner product on unit vectors)
    index = faiss.IndexFlatIP(dim)  # inner product
    ids: List[str] = []
    all_vecs: List[np.ndarray] = []

    _log(f"Embedding {len(df)} chunks with batch={batch}, dim={dim}")
    for part in chunk_iter(df[["chunk_id", "text"]], batch):
        texts = part["text"].astype(str).tolist()
        vecs = emb.embed(texts, batch_size=batch)
        if vecs.shape[1] != dim:
            raise RuntimeError(
                f"Embedding dim mismatch: got {vecs.shape[1]}, expected {dim}")
        vecs = l2_normalize(vecs)  # cosine
        index.add(vecs)
        all_vecs.append(vecs)
        ids.extend(part["chunk_id"].tolist())

    # Persist
    embs = np.vstack(all_vecs).astype(np.float32, copy=False)

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    np.save(DATA_DIR / "embeddings.npy", embs)
    np.save(DATA_DIR / "chunk_ids.npy", np.array(ids, dtype=object))

    # Save minimal meta frame for retriever joins (fast to load)
    meta_cols = ["chunk_id", "doc_id", "observed_at",
                 "source_path", "published_at", "valid_from", "valid_to", "tz"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].copy()
    meta.to_parquet(DATA_DIR / "chunk_meta.parquet", index=False)

    faiss.write_index(index, str(index_out))

    _log(f"Index written: {index_out}")
    _log(f"Embeddings: {DATA_DIR/'embeddings.npy'} shape={embs.shape}")
    _log(f"IDs:         {DATA_DIR/'chunk_ids.npy'} count={len(ids)}")
    _log(f"Meta:        {DATA_DIR/'chunk_meta.parquet'} rows={len(meta)}")

# ---------------------------
# CLI
# ---------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ingest.build_faiss",
        description="Build FAISS cosine index from data/chunks.csv",
    )
    p.add_argument("--chunks", type=str, default=str(DATA_DIR /
                   "chunks.csv"), help="Path to chunks.csv")
    p.add_argument("--model", type=str,
                   default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    p.add_argument("--dim", type=int, default=None,
                   help="Override embedding dim (optional)")
    p.add_argument("--batch", type=int, default=64,
                   help="Batch size for embedding")
    p.add_argument("--prefer_fastembed", action="store_true",
                   help="Use fastembed if available")
    p.add_argument("--index_out", type=str, default=str(INDEX_DIR /
                   "faiss.index"), help="Output FAISS index path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    build_index(
        chunks_csv=Path(args.chunks),
        model_name=args.model,
        dim=args.dim,
        batch=args.batch,
        prefer_fastembed=args.prefer_fastembed,
        index_out=Path(args.index_out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
