#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — FastAPI surface
Routes (spec-aligned):
  • POST /api/v1/retrieve → retrieval-only (time routing + Hop-1 + DHQC decision)
  • POST /api/v1/answer   → retrieval + (generator or Lite fallback) + attribution (if available)
  • GET  /api/v1/health   → liveness

No deviations from PoC; /ingest remains a CLI task (see ingest.loader + ingest.build_faiss).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.orchestrator import (
    QueryRequest,
    retrieve_only,
    answer as answer_fn,
    EvidenceSnippet,
    ControllerStats,
    TimeFilters,
)

app = FastAPI(title="ChronoRAG-Lite API", version="1.0")

# CORS (open by default for local dev; tighten in PRO)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Pydantic I/O models (mirror orchestrator dataclasses exactly)
# ------------------------------------------------------------------------------


class APIQuery(BaseModel):
    query: str = Field(..., description="Natural language question")
    k: int = 8
    time_mode: Optional[str] = Field(
        None, description="INTELLIGENT | HARD | None")
    time_hint: Optional[Dict[str, Any]] = None
    fiscal_start_month: int = 4
    model_name: str = "BAAI/bge-small-en-v1.5"
    prefer_fastembed: bool = True


class APISnippet(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score_total: float
    score_sim: float
    score_time: float
    score_age: float
    source_path: Optional[str] = None
    observed_at: Optional[str] = None
    published_at: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    tz: Optional[str] = None


class APIControllerSignals(BaseModel):
    coverage: float
    uncertainty: float
    contradiction: float
    temporal_valid: float
    predicted_gain: float
    budget_slack: float


class APIControllerStats(BaseModel):
    mode: str
    tau: float
    delta: float
    n_max: int
    hops_used: int
    decision: str
    latency_ms: Optional[int] = None
    notes: List[str] = []
    signals_hop1: APIControllerSignals


class APITimeFilters(BaseModel):
    mode: str
    confidence: float
    start: Optional[str]
    end: Optional[str]


class APIRetrievalResponse(BaseModel):
    query: str
    time_filters: APITimeFilters
    candidates: List[APISnippet]
    controller_stats: APIControllerStats


class APIAnswerResponse(BaseModel):
    query: str
    answer: str
    time_filters: APITimeFilters
    candidates: List[APISnippet]
    controller_stats: APIControllerStats
    attribution_card: Optional[Dict[str, Any]] = None

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def _to_query_request(body: APIQuery) -> QueryRequest:
    return QueryRequest(
        query=body.query,
        k=body.k,
        time_mode=body.time_mode,
        time_hint=body.time_hint,
        fiscal_start_month=body.fiscal_start_month,
        model_name=body.model_name,
        prefer_fastembed=body.prefer_fastembed,
    )


def _to_api_snippets(snips: List[EvidenceSnippet]) -> List[APISnippet]:
    return [APISnippet(**asdict(s)) for s in snips]


def _to_api_stats(stats: ControllerStats) -> APIControllerStats:
    sig = stats.signals_hop1
    return APIControllerStats(
        mode=stats.mode,
        tau=stats.tau,
        delta=stats.delta,
        n_max=stats.n_max,
        hops_used=stats.hops_used,
        decision=stats.decision,
        latency_ms=stats.latency_ms,
        notes=stats.notes,
        signals_hop1=APIControllerSignals(
            coverage=sig.coverage,
            uncertainty=sig.uncertainty,
            contradiction=sig.contradiction,
            temporal_valid=sig.temporal_valid,
            predicted_gain=sig.predicted_gain,
            budget_slack=sig.budget_slack,
        ),
    )


def _to_api_tf(tf: TimeFilters) -> APITimeFilters:
    return APITimeFilters(**asdict(tf))

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------


@app.get("/api/v1/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/retrieve", response_model=APIRetrievalResponse)
def api_retrieve(body: APIQuery) -> APIRetrievalResponse:
    req = _to_query_request(body)
    out = retrieve_only(req)
    return APIRetrievalResponse(
        query=out.query,
        time_filters=_to_api_tf(out.time_filters),
        candidates=_to_api_snippets(out.candidates),
        controller_stats=_to_api_stats(out.controller_stats),
    )


@app.post("/api/v1/answer", response_model=APIAnswerResponse)
def api_answer(body: APIQuery) -> APIAnswerResponse:
    req = _to_query_request(body)
    out = answer_fn(req)
    return APIAnswerResponse(
        query=out.query,
        answer=out.answer,
        time_filters=_to_api_tf(out.time_filters),
        candidates=_to_api_snippets(out.candidates),
        controller_stats=_to_api_stats(out.controller_stats),
        attribution_card=out.attribution_card,
    )
