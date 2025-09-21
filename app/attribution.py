#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Attribution Cards (ChronoChips)
Spec-aligned:
  • Compact badges (mode, window, confidence)
  • Evidence receipts (top snippets with windows & source)
  • Controller trace (S1 notes, decision)
  • Time filters echoed verbatim for audit

Input contracts match orchestrator.AnswerResponse builder.
"""

from __future__ import annotations
from typing import Any, Dict, List


def _badge(label: str, value: str) -> Dict[str, str]:
    return {"label": label, "value": value}


def _safe(s: Any) -> str:
    return "" if s is None else str(s)


def build_attribution_card(
    query: str,
    candidates: List[Any],                 # retriever.Candidate list
    # {"mode","confidence","start","end"}
    time_filters: Dict[str, Any],
    mode: str,
    controller_stats: Dict[str, Any],      # asdict(ControllerStats)
) -> Dict[str, Any]:
    # Badges (top row)
    badges = [
        _badge("MODE", _safe(time_filters.get("mode")) or _safe(mode)),
        _badge(
            "WINDOW", f"[{_safe(time_filters.get('start'))} .. {_safe(time_filters.get('end'))})"),
        _badge("CONF", f"{float(time_filters.get('confidence', 0.0)):.2f}"),
        _badge("HOPS", str(controller_stats.get("hops_used", 1))),
        _badge("DECISION", _safe(controller_stats.get("decision"))),
    ]

    # Evidence receipts
    receipts = []
    for i, c in enumerate(candidates[:8], 1):
        vf = getattr(c, "valid_from", None)
        vt = getattr(c, "valid_to", None)
        pub = getattr(c, "published_at", None)
        window = f"[{vf or pub} .. {(vt or vf or pub)})" if (
            vf or vt or pub) else "[unknown]"
        receipts.append({
            "rank": i,
            "chunk_id": getattr(c, "chunk_id", ""),
            "doc_id": getattr(c, "doc_id", ""),
            "window": window,
            "source": getattr(c, "source_path", None),
            "scores": {
                "total": round(float(getattr(c, "score_total", 0.0)), 3),
                "sim":   round(float(getattr(c, "score_sim", 0.0)), 3),
                "time":  round(float(getattr(c, "score_time", 0.0)), 3),
                "age":   round(float(getattr(c, "score_age", 0.0)), 3),
            },
            "quote": (getattr(c, "text", "") or "").replace("\n", " ")[:400],
        })

    # Controller trace (compact)
    sig = controller_stats.get("signals_hop1", {}) or {}
    trace = {
        "tau": controller_stats.get("tau"),
        "delta": controller_stats.get("delta"),
        "n_max": controller_stats.get("n_max"),
        "signals": {
            "coverage":      round(float(sig.get("coverage", 0.0)), 3),
            "uncertainty":   round(float(sig.get("uncertainty", 0.0)), 3),
            "contradiction": round(float(sig.get("contradiction", 0.0)), 3),
            "temporal_valid": round(float(sig.get("temporal_valid", 0.0)), 3),
            "predicted_gain": round(float(sig.get("predicted_gain", 0.0)), 3),
            "budget_slack":  round(float(sig.get("budget_slack", 0.0)), 3),
        },
        "notes": controller_stats.get("notes", []),
    }

    return {
        "query": query,
        "badges": badges,
        "time_filters": time_filters,
        "receipts": receipts,
        "controller": trace,
        # Future fields kept for parity with full ChronoCards:
        # e.g., overlapping claims (out-of-scope in Lite)
        "conflicts": [],
        "counterfactuals": [],  # near-miss evidence (optional in Lite)
        "policy": {            # POLAR echo (Lite placeholders)
            "alpha": 0.55, "beta_time": 0.25, "gamma_authority": 0.15, "delta_age": 0.05
        },
        "version": "chrono-lite-1.0",
    }
