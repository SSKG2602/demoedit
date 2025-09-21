#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Retriever
Responsibilities
  • Load persisted FAISS artifacts (indexes/faiss.index, data/chunk_meta.parquet)
  • Embed queries with the SAME model used at build time
  • Perform cosine ANN search
  • Apply time filters (INTELLIGENT/HARD handled upstream; here we accept windows)
  • Compute time-aware fusion score (α·sim + β·time − δ·age [+ γ·authority placeholder])
  • (Optional) Hook for Neo4j graph expansion (kept off by default to stay Lite)

Design Notes
  • Low-RAM: lazy loads, reuses embedder, avoids copying large arrays
  • Deterministic & debug-friendly: rich logs, explicit dataclasses
  • Reranker is OFF by default (PoC: optional)

Public API
  • RetrieverConfig — tunables aligned with PoC weights
  • Retriever.load(...)  — factory from disk
  • Retriever.search(query, k, time_window=..., mode=...) -> RetrievalResult
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

# ---------------------------
# Logging
# ---------------------------


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[retriever {ts}] {msg}", flush=True)

# ---------------------------
# Embedding provider (same contract as build_faiss)
# ---------------------------


class Embedder:
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class FastEmbedder(Embedder):
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # type: ignore
        self.model = TextEmbedding(model_name=model_name)
        try:
            self._dim = self.model.get_output_dim()  # type: ignore
        except Exception:
            self._dim = len(list(self.model.embed(["probe"]))[0])

    @property
    def dim(self) -> int:
        return int(self._dim)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embs = list(self.model.embed(texts))
        return np.asarray(embs, dtype=np.float32)


class STEmbedder(Embedder):
    def __init__(self, model_name: str, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name, device=device or "cpu")
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
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False
        )
        return vecs.astype(np.float32, copy=False)


def resolve_embedder(model_name: str, prefer_fastembed: bool = True) -> Embedder:
    if prefer_fastembed:
        try:
            _log(f"embedder=fastembed model={model_name}")
            return FastEmbedder(model_name)
        except Exception as e:
            _log(
                f"fastembed unavailable → fallback to sentence-transformers ({e})")
    _log(f"embedder=sentence-transformers model={model_name}")
    return STEmbedder(model_name)

# ---------------------------
# Data models
# ---------------------------


@dataclass
class TimeWindow:
    start: Optional[datetime] = None  # inclusive
    end: Optional[datetime] = None    # exclusive (PoC '[)' style)
    # For Lite we accept naive UTC; upstream temporal module sets these.


@dataclass
class Candidate:
    chunk_id: str
    doc_id: str
    text: str
    score_sim: float
    score_time: float
    score_age: float
    score_total: float
    observed_at: Optional[str] = None
    published_at: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    tz: Optional[str] = None
    source_path: Optional[str] = None
    rank: int = -1


@dataclass
class RetrievalResult:
    query: str
    k: int
    time_window: Optional[TimeWindow]
    mode: str  # "INTELLIGENT" or "HARD"
    candidates: List[Candidate] = field(default_factory=list)

# ---------------------------
# Config
# ---------------------------


@dataclass
class RetrieverConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    prefer_fastembed: bool = True
    k0: int = 50                        # ANN fan-in before fusion
    alpha: float = 0.55                 # sim weight
    beta_time: float = 0.25             # time overlap/decay
    delta_age: float = 0.05             # age penalty
    # placeholder (kept 0 in Lite unless you pass extras)
    gamma_authority: float = 0.15
    recency_halflife_days: float = 350.0  # λ ~ 0.002 ~ 350 days half-life
    distance_mu: float = 0.01           # soft decay for outside-window evidence

# ---------------------------
# Core Retriever
# ---------------------------


class Retriever:
    def __init__(
        self,
        index: faiss.Index,
        ids: np.ndarray,
        meta: pd.DataFrame,
        embedder: Embedder,
        cfg: RetrieverConfig,
    ) -> None:
        self.index = index
        self.ids = ids  # dtype=object array of chunk_id
        self.meta = meta.set_index("chunk_id", drop=False)
        self.embedder = embedder
        self.cfg = cfg

        # Sanity checks
        if not self.index.is_trained:
            _log("WARNING: FAISS index reports untrained (IndexFlatIP is OK)")
        _log(
            f"retriever ready: dim={self.embedder.dim}, n={self.index.ntotal}")

    @staticmethod
    def load(
        index_path: Path = Path("indexes/faiss.index"),
        ids_path: Path = Path("data/chunk_ids.npy"),
        meta_path: Path = Path("data/chunk_meta.parquet"),
        model_name: str = "BAAI/bge-small-en-v1.5",
        prefer_fastembed: bool = True,
        cfg: Optional[RetrieverConfig] = None,
    ) -> "Retriever":
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index: {index_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing IDs: {ids_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata: {meta_path}")

        index = faiss.read_index(str(index_path))
        ids = np.load(ids_path, allow_pickle=True)
        meta = pd.read_parquet(meta_path)

        embedder = resolve_embedder(
            model_name, prefer_fastembed=prefer_fastembed)
        cfg = cfg or RetrieverConfig(
            model_name=model_name, prefer_fastembed=prefer_fastembed)
        return Retriever(index=index, ids=ids, meta=meta, embedder=embedder, cfg=cfg)

    # ---------- scoring helpers ----------
    @staticmethod
    def _parse_rfc3339(s: Optional[str]) -> Optional[datetime]:
        if not s or not isinstance(s, str):
            return None
        try:
            # Accept 'Z' or offset-naive; coerce to UTC
            if s.endswith("Z"):
                return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
            return datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            return None

    def _time_weight(self, row: pd.Series, window: Optional[TimeWindow]) -> float:
        """
        Soft time weight (INTELLIGENT): overlap IoU if both sides have windows,
        else distance decay from center. HARD is handled upstream by masking;
        here we reuse the same function but expect candidates to be pre-masked.
        """
        if window is None:
            return 0.0

        # Evidence window: prefer valid_from/to, else fallback to published_at
        ev_start = self._parse_rfc3339(
            row.get("valid_from")) or self._parse_rfc3339(row.get("published_at"))
        ev_end = self._parse_rfc3339(row.get("valid_to")) or ev_start

        if ev_start and ev_end and window.start and window.end:
            # IoU on [start, end) in seconds
            a0, a1 = ev_start.timestamp(), ev_end.timestamp()
            b0, b1 = window.start.timestamp(), window.end.timestamp()
            if a1 <= a0 or b1 <= b0:
                return 0.0
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            union = (a1 - a0) + (b1 - b0) - inter
            iou = inter / max(union, 1e-9)
            # small edge ramp if touching but not overlapping
            if inter == 0.0:
                # distance (seconds) to nearest edge
                dist = min(abs(a0 - b1), abs(b0 - a1))
                days = dist / 86400.0
                return float(np.exp(-self.cfg.distance_mu * days))
            return float(iou)

        # Fallback: distance from observed_at to window center
        obs = self._parse_rfc3339(row.get("observed_at"))
        if obs and (window.start or window.end):
            center = window.start or window.end
            days = abs((obs - center).total_seconds()) / 86400.0
            return float(np.exp(-self.cfg.distance_mu * days))

        return 0.0

    def _age_penalty(self, row: pd.Series) -> float:
        """
        Recency decay based on 'observed_at' (Lite approximation).
        penalty ∈ [0, 1]; higher = more penalty.
        """
        obs = self._parse_rfc3339(row.get("observed_at"))
        if not obs:
            return 0.0
        days = (datetime.now(timezone.utc) - obs).total_seconds() / 86400.0
        # half-life → convert to penalty in [0,1]
        lam = np.log(2) / max(self.cfg.recency_halflife_days, 1.0)
        # convert decay to penalty (older → closer to 1.0)
        return float(1.0 - np.exp(-lam * days))

    # ---------- main API ----------
    def search(
        self,
        query: str,
        k: int = 8,
        time_window: Optional[TimeWindow] = None,
        # or "HARD" (HARD: upstream should mask; we still compute time weight)
        mode: str = "INTELLIGENT",
        # future: authority scores, graph paths, etc.
        extras: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        # Embed query and L2-normalize (cosine on IP index)
        qv = self.embedder.embed([query], batch_size=1).astype(np.float32)
        qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

        # ANN search
        k0 = max(self.cfg.k0, k)
        sims, idxs = self.index.search(qv, k0)  # shapes: (1,k0)
        sims = sims[0].astype(np.float32)
        idxs = idxs[0].astype(np.int32)

        # Assemble candidates with fusion scoring
        rows: List[Candidate] = []
        for rank0, (score_sim, idx) in enumerate(zip(sims, idxs)):
            if idx < 0 or idx >= len(self.ids):
                continue
            chunk_id = str(self.ids[idx])
            row = self.meta.loc[chunk_id]

            # HARD mode masking (if upstream didn't filter, mask here conservatively)
            if mode.upper() == "HARD" and time_window is not None:
                v_from = self._parse_rfc3339(
                    row.get("valid_from")) or self._parse_rfc3339(row.get("published_at"))
                v_to = self._parse_rfc3339(row.get("valid_to")) or v_from
                if v_from and v_to and (time_window.start and time_window.end):
                    # require strict overlap (BETWEEN semantics)
                    if not (v_from < time_window.end and v_to > time_window.start):
                        continue  # outside → drop

            tw = self._time_weight(row, time_window)
            ap = self._age_penalty(row)
            # authority placeholder
            auth = float(extras.get("authority", 0.0)) if extras else 0.0

            total = (
                self.cfg.alpha * float(score_sim) +
                self.cfg.beta_time * tw +
                self.cfg.gamma_authority * auth -
                self.cfg.delta_age * ap
            )

            rows.append(Candidate(
                chunk_id=chunk_id,
                doc_id=str(row.get("doc_id")),
                text=str(row.get("text", "")),
                score_sim=float(score_sim),
                score_time=float(tw),
                score_age=float(ap),
                score_total=float(total),
                observed_at=str(row.get("observed_at")) if pd.notna(
                    row.get("observed_at")) else None,
                published_at=str(row.get("published_at")) if pd.notna(
                    row.get("published_at")) else None,
                valid_from=str(row.get("valid_from")) if pd.notna(
                    row.get("valid_from")) else None,
                valid_to=str(row.get("valid_to")) if pd.notna(
                    row.get("valid_to")) else None,
                tz=str(row.get("tz")) if pd.notna(row.get("tz")) else None,
                source_path=str(row.get("source_path")) if pd.notna(
                    row.get("source_path")) else None,
                rank=rank0,
            ))

        # Sort by fused score and truncate to k
        rows.sort(key=lambda c: c.score_total, reverse=True)
        rows = rows[:k]

        return RetrievalResult(
            query=query, k=k, time_window=time_window, mode=mode.upper(), candidates=rows
        )

# ---------------------------
# Quick CLI (debug aid)
# ---------------------------


def _parse_cli_timewindow(s: Optional[str]) -> Optional[TimeWindow]:
    """
    Accept 'YYYY-MM-DD..YYYY-MM-DD' (end exclusive), or single 'YYYY-MM-DD' (AS_OF day).
    """
    if not s:
        return None
    try:
        if ".." in s:
            a, b = s.split("..", 1)
            A = datetime.fromisoformat(a).replace(tzinfo=timezone.utc)
            B = datetime.fromisoformat(b).replace(tzinfo=timezone.utc)
            return TimeWindow(start=A, end=B)
        A = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        B = A
        return TimeWindow(start=A, end=B)
    except Exception:
        return None


def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="ChronoRAG-Lite retriever (debug CLI)")
    p.add_argument("--query", required=True, help="natural language query")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--time", type=str, default=None,
                   help="YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD")
    p.add_argument("--mode", type=str, default="INTELLIGENT",
                   choices=["INTELLIGENT", "HARD"])
    p.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5")
    p.add_argument("--prefer_fastembed", action="store_true")
    args = p.parse_args()

    ret = Retriever.load(
        model_name=args.model,
        prefer_fastembed=args.prefer_fastembed,
    )
    tw = _parse_cli_timewindow(args.time)
    out = ret.search(query=args.query, k=args.k,
                     time_window=tw, mode=args.mode)

    print(
        f"\nTop-{args.k} for: {args.query!r} | mode={args.mode} | window={args.time or 'None'}\n")
    for i, c in enumerate(out.candidates, 1):
        print(f"{i:02d}. total={c.score_total:.3f}  sim={c.score_sim:.3f}  time={c.score_time:.3f}  age={c.score_age:.3f}")
        print(f"    {c.text[:140].replace('\\n',' ')}")
        print(
            f"    chunk_id={c.chunk_id}  doc_id={c.doc_id}  src={c.source_path}\n")


if __name__ == "__main__":
    _cli()
