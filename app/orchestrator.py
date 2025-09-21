#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Orchestrator (DHQC 1+N, Lite)
Spec alignment:
  • Temporal routing → Hop-1 retrieve → SIGMA-6 signals → QPLAN-Δ (static) early-exit
  • PRO defaults (deterministic): tau=0.80, delta=0.20, n_max=3
  • Evidence & time filters are returned for UI/Attribution; generator is optional (Lite-safe fallback)

Public entry points:
  • retrieve_only(req: QueryRequest) -> RetrievalResponse
  • answer(req: QueryRequest) -> AnswerResponse

This file is intentionally verbose with comments/logs for debuggability on macOS + conda.
No deviations from the architecture; placeholders are clearly labeled.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np

from app.temporal import normalize_query_time, Selection, TimeWindow
from app.retriever import Retriever, RetrieverConfig, Candidate, RetrievalResult

# Optional: generator & attribution (Lite-safe, import guarded)
try:
    # expected signature: (query, candidates, time_filters, mode) -> str
    from app.synthesize import generate_answer
except Exception:  # pragma: no cover
    generate_answer = None  # fallback later

try:
    # expected signature: (...)
    from app.attribution import build_attribution_card
except Exception:  # pragma: no cover
    build_attribution_card = None

# -----------------------------------------------------------------------------
# Configuration (POLAR-like snippet for Lite)
# -----------------------------------------------------------------------------
TAU: float = 0.80             # sufficiency target
DELTA: float = 0.20           # static quantizer step
N_MAX: int = 3                # cap extra hops

DEFAULT_K: int = 8            # final k for the caller
# ann fan-in (also exists in RetrieverConfig; kept here for clarity)
K0: int = 50
MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
PREFER_FASTEMBED: bool = True

# -----------------------------------------------------------------------------
# Data Models (API-ready shapes for FastAPI later)
# -----------------------------------------------------------------------------


@dataclass
class QueryRequest:
    query: str
    k: int = DEFAULT_K
    # "INTELLIGENT" | "HARD" | None → router decides
    time_mode: Optional[str] = None
    # e.g., {"operator":"BETWEEN","from":"2025-04-01","to":"2025-06-30"}
    time_hint: Optional[Dict[str, Any]] = None
    fiscal_start_month: int = 4
    # knobs
    model_name: str = MODEL_NAME
    prefer_fastembed: bool = PREFER_FASTEMBED


@dataclass
class ControllerSignals:
    coverage: float
    uncertainty: float
    contradiction: float
    temporal_valid: float
    predicted_gain: float
    budget_slack: float


@dataclass
class ControllerStats:
    mode: str
    tau: float
    delta: float
    n_max: int
    hops_used: int
    signals_hop1: ControllerSignals
    decision: str  # "early_exit" | "escalate" | "budget_exhausted"
    latency_ms: Optional[int] = None  # filled by API layer if desired
    notes: List[str] = field(default_factory=list)


@dataclass
class TimeFilters:
    mode: str
    confidence: float
    start: Optional[str]
    end: Optional[str]


@dataclass
class EvidenceSnippet:
    chunk_id: str
    doc_id: str
    text: str
    score_total: float
    score_sim: float
    score_time: float
    score_age: float
    source_path: Optional[str]
    observed_at: Optional[str]
    published_at: Optional[str]
    valid_from: Optional[str]
    valid_to: Optional[str]
    tz: Optional[str]

    @staticmethod
    def from_candidate(c: Candidate) -> "EvidenceSnippet":
        return EvidenceSnippet(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            text=c.text,
            score_total=c.score_total,
            score_sim=c.score_sim,
            score_time=c.score_time,
            score_age=c.score_age,
            source_path=c.source_path,
            observed_at=c.observed_at,
            published_at=c.published_at,
            valid_from=c.valid_from,
            valid_to=c.valid_to,
            tz=c.tz,
        )


@dataclass
class RetrievalResponse:
    query: str
    time_filters: TimeFilters
    candidates: List[EvidenceSnippet]
    controller_stats: ControllerStats


@dataclass
class AnswerResponse:
    query: str
    answer: str
    time_filters: TimeFilters
    candidates: List[EvidenceSnippet]
    controller_stats: ControllerStats
    attribution_card: Optional[Dict[str, Any]] = None

# -----------------------------------------------------------------------------
# Helpers — SIGMA-6 (Lite estimates)
# -----------------------------------------------------------------------------


def _estimate_coverage(cands: List[Candidate]) -> float:
    """
    Lite proxy: coverage ~ diminishing return over top-k diversity.
    Here we approximate with normalized sum of total scores.
    """
    if not cands:
        return 0.0
    scores = np.array([max(0.0, c.score_total)
                      for c in cands], dtype=np.float32)
    s = float(scores.sum())
    # map to 0..1 with soft saturation
    return float(1.0 - np.exp(-s / max(1e-6, scores.size)))


def _estimate_uncertainty(cands: List[Candidate]) -> float:
    """
    Uncertainty ~ score dispersion (higher dispersion → lower uncertainty).
    We invert a normalized stddev; clamp to [0,1].
    """
    if len(cands) < 2:
        return 1.0
    xs = np.array([c.score_total for c in cands], dtype=np.float32)
    xs = (xs - xs.min()) / (xs.ptp() + 1e-6)
    std = float(xs.std())
    # high std → more separation → LOWER uncertainty
    return float(max(0.0, 1.0 - std))


def _estimate_contradiction(_: List[Candidate]) -> float:
    """
    Placeholder (Lite): we don’t run claim graph; keep neutral 0.0.
    """
    return 0.0


def _estimate_temporal_valid(cands: List[Candidate]) -> float:
    """
    Temporal validity ~ average time weight from retriever (already in 0..1).
    """
    if not cands:
        return 0.0
    return float(np.mean([max(0.0, c.score_time) for c in cands]))


def _estimate_predicted_gain(_: List[Candidate]) -> float:
    """
    Placeholder (Lite): GSM-GainPredict is out-of-scope here; keep small positive hint.
    """
    return 0.05


def _budget_slack(hops_used: int, n_max: int) -> float:
    """
    Slack ~ proportion of budget left (0..1).
    """
    return float(max(0, n_max - hops_used) / max(1, n_max))


def _sufficiency(sig: ControllerSignals) -> float:
    """
    S1 = σ(w·[C, 1-U, 1-K, T, G, B] + b) simplified:
    We’ll use an uncalibrated affine combo mapped to [0,1] via squashing.
    """
    C = sig.coverage
    U = sig.uncertainty
    K = sig.contradiction
    T = sig.temporal_valid
    G = sig.predicted_gain
    B = sig.budget_slack
    raw = 0.35*C + 0.20*(1.0-U) + 0.10*(1.0-K) + 0.25*T + 0.05*G + 0.05*B
    # simple squash (keep monotone, bounded)
    return float(max(0.0, min(1.0, raw)))

# -----------------------------------------------------------------------------
# Core steps
# -----------------------------------------------------------------------------


def _time_filters_from_selection(sel: Selection) -> TimeFilters:
    s = sel.window.start.isoformat() if sel.window and sel.window.start else None
    e = sel.window.end.isoformat() if sel.window and sel.window.end else None
    return TimeFilters(mode=sel.mode, confidence=sel.confidence, start=s, end=e)


def _to_evidence_snippets(cands: List[Candidate]) -> List[EvidenceSnippet]:
    return [EvidenceSnippet.from_candidate(c) for c in cands]


def _compute_signals(hops_used: int, cands: List[Candidate]) -> ControllerSignals:
    sig = ControllerSignals(
        coverage=_estimate_coverage(cands),
        uncertainty=_estimate_uncertainty(cands),
        contradiction=_estimate_contradiction(cands),
        temporal_valid=_estimate_temporal_valid(cands),
        predicted_gain=_estimate_predicted_gain(cands),
        budget_slack=_budget_slack(hops_used, N_MAX),
    )
    return sig


def _dhqc_decide(sig: ControllerSignals, tau: float = TAU, delta: float = DELTA, n_max: int = N_MAX) -> Tuple[str, int, float]:
    """
    Static quantizer (QPLAN-Δ): map sufficiency gap to N extra hops.
    Returns (decision, n_more, S1).
    """
    S1 = _sufficiency(sig)
    if S1 >= tau:
        return "early_exit", 0, S1
    gap = max(0.0, tau - S1)
    n = int(gap // delta) + (1 if gap > 0 else 0)
    n = max(0, min(n, n_max))
    if n == 0:
        return "early_exit", 0, S1
    return "escalate", n, S1

# -----------------------------------------------------------------------------
# Orchestrator (retrieval-only and answer)
# -----------------------------------------------------------------------------


def _ensure_retriever(cfg: RetrieverConfig) -> Retriever:
    return Retriever.load(
        model_name=cfg.model_name,
        prefer_fastembed=cfg.prefer_fastembed,
    )


def retrieve_only(req: QueryRequest) -> RetrievalResponse:
    # 1) Temporal routing
    sel = normalize_query_time(
        req.query, fiscal_start_month=req.fiscal_start_month)
    if req.time_mode and req.time_mode.upper() in ("HARD", "INTELLIGENT"):
        sel.mode = req.time_mode.upper()  # manual override if provided

    # 2) Hop-1 retrieve
    ret_cfg = RetrieverConfig(
        model_name=req.model_name,
        prefer_fastembed=req.prefer_fastembed,
        k0=K0,
    )
    retriever = _ensure_retriever(ret_cfg)

    r1: RetrievalResult = retriever.search(
        query=req.query,
        k=req.k,
        time_window=sel.window,
        mode=sel.mode,
    )

    # 3) Signals + decision
    sig1 = _compute_signals(hops_used=1, cands=r1.candidates)
    decision, n_more, S1 = _dhqc_decide(sig1)

    notes: List[str] = [f"S1={S1:.3f}",
                        f"decision={decision}", f"n_more={n_more}"]
    stats = ControllerStats(
        mode=sel.mode,
        tau=TAU,
        delta=DELTA,
        n_max=N_MAX,
        hops_used=1,
        signals_hop1=sig1,
        decision=decision,
        notes=notes,
    )

    # Lite: do not actually run extra hops; we preserve decision & notes for observability.
    return RetrievalResponse(
        query=req.query,
        time_filters=_time_filters_from_selection(sel),
        candidates=_to_evidence_snippets(r1.candidates),
        controller_stats=stats,
    )


def _fallback_generate(query: str, cands: List[Candidate], tf: TimeFilters, mode: str) -> str:
    """
    Minimal, deterministic fallback generator (no LLM).
    """
    head = f"Answer (Lite): based on top-{min(3, len(cands))} evidence; mode={mode}, window=[{tf.start}..{tf.end})"
    bullets = []
    for c in cands[:3]:
        src = c.source_path or c.doc_id
        bullets.append(f"- {c.text[:180].replace(chr(10),' ')}  (src={src})")
    return head + "\n" + "\n".join(bullets)


def answer(req: QueryRequest) -> AnswerResponse:
    # 1) Temporal routing
    sel = normalize_query_time(
        req.query, fiscal_start_month=req.fiscal_start_month)
    if req.time_mode and req.time_mode.upper() in ("HARD", "INTELLIGENT"):
        sel.mode = req.time_mode.upper()

    # 2) Hop-1 retrieve
    ret_cfg = RetrieverConfig(
        model_name=req.model_name,
        prefer_fastembed=req.prefer_fastembed,
        k0=K0,
    )
    retriever = _ensure_retriever(ret_cfg)
    r1: RetrievalResult = retriever.search(
        query=req.query,
        k=req.k,
        time_window=sel.window,
        mode=sel.mode,
    )

    # 3) DHQC signals/decision
    sig1 = _compute_signals(hops_used=1, cands=r1.candidates)
    decision, n_more, S1 = _dhqc_decide(sig1)

    notes: List[str] = [f"S1={S1:.3f}",
                        f"decision={decision}", f"n_more={n_more}"]
    stats = ControllerStats(
        mode=sel.mode,
        tau=TAU,
        delta=DELTA,
        n_max=N_MAX,
        hops_used=1,
        signals_hop1=sig1,
        decision=decision,
        notes=notes,
    )

    # 4) Generate answer (LLM or fallback)
    time_filters = _time_filters_from_selection(sel)
    if generate_answer is not None:
        try:
            text = generate_answer(
                query=req.query,
                candidates=r1.candidates,
                time_filters=asdict(time_filters),
                mode=sel.mode,
            )
        except Exception:
            text = _fallback_generate(
                req.query, r1.candidates, time_filters, sel.mode)
    else:
        text = _fallback_generate(
            req.query, r1.candidates, time_filters, sel.mode)

    # 5) Attribution card (optional)
    card = None
    if build_attribution_card is not None:
        try:
            card = build_attribution_card(
                query=req.query,
                candidates=r1.candidates,
                time_filters=asdict(time_filters),
                mode=sel.mode,
                controller_stats=asdict(stats),
            )
        except Exception:
            card = None

    return AnswerResponse(
        query=req.query,
        answer=text,
        time_filters=time_filters,
        candidates=_to_evidence_snippets(r1.candidates),
        controller_stats=stats,
        attribution_card=card,
    )

# -----------------------------------------------------------------------------
# CLI (quick smoke test)
# -----------------------------------------------------------------------------


def _print_retrieval(resp: RetrievalResponse) -> None:
    print(f"\nQUERY: {resp.query}")
    tf = resp.time_filters
    print(
        f"MODE={tf.mode}  CONF={tf.confidence:.2f}  WINDOW=[{tf.start} .. {tf.end})")
    print(f"HOPS_USED={resp.controller_stats.hops_used}  DECISION={resp.controller_stats.decision}  NOTES={resp.controller_stats.notes}")
    for i, c in enumerate(resp.candidates, 1):
        print(f"{i:02d}. total={c.score_total:.3f} sim={c.score_sim:.3f} time={c.score_time:.3f} age={c.score_age:.3f}")
        print(f"    {c.text[:140].replace(chr(10),' ')}")
        print(
            f"    src={c.source_path}  vf={c.valid_from}  vt={c.valid_to}  obs={c.observed_at}")


def _print_answer(resp: AnswerResponse) -> None:
    print(f"\nQUERY: {resp.query}")
    print(
        f"MODE={resp.time_filters.mode}  WINDOW=[{resp.time_filters.start} .. {resp.time_filters.end})")
    print(
        f"HOPS_USED={resp.controller_stats.hops_used}  DECISION={resp.controller_stats.decision}")
    print("\n--- ANSWER ---")
    print(resp.answer)
    print("\n--- TOP EVIDENCE ---")
    for i, c in enumerate(resp.candidates[:3], 1):
        print(
            f"{i:02d}. {c.text[:160].replace(chr(10),' ')}  (src={c.source_path})")


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="ChronoRAG-Lite Orchestrator CLI")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument(
        "--action", choices=["retrieve", "answer"], default="retrieve")
    ap.add_argument("--time_mode", type=str, default=None,
                    choices=[None, "INTELLIGENT", "HARD"])
    args = ap.parse_args()

    req = QueryRequest(query=args.query, k=args.k, time_mode=args.time_mode)
    if args.action == "retrieve":
        out = retrieve_only(req)
        _print_retrieval(out)
    else:
        out = answer(req)
        _print_answer(out)


if __name__ == "__main__":
    _cli()
