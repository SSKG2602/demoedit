#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Temporal Normalizer & Router
Spec-aligned (no deviations):
  • INTELLIGENT: propose candidates → score evidence density upstream; here we just parse + score confidence heuristics.
  • HARD: explicit windows (AS OF, BETWEEN, Qx FYyy) → strict interval.
  • Default TZ handling: produce timezone-aware UTC datetimes; store strings as RFC-3339 if needed upstream.

⚠️ Quarter mapping follows your doc's worked example:
    Fiscal start = April  →  Q2 FY25 == 2025-04-01 .. 2025-06-30   (HARD)
This keeps Q2 ≡ Apr–Jun, Q3 ≡ Jul–Sep, Q4 ≡ Oct–Dec, Q1 ≡ Jan–Mar (of the same FY label+1).

Public API:
  - TimeWindow(start,end)  # end exclusive
  - Selection(window, mode, confidence, trace)
  - normalize_query_time(query, now_utc=None, fiscal_start_month=4) -> Selection

CLI:
  python -m app.temporal --q "revenue in Q2 FY25"
  python -m app.temporal --q "as of 2025-08-15 who is CEO?"
  python -m app.temporal --q "last 90 days"
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict

RFC3339 = "%Y-%m-%dT%H:%M:%S%z"

# ---------------------------
# Data types
# ---------------------------


@dataclass
class TimeWindow:
    start: Optional[datetime] = None  # inclusive
    end: Optional[datetime] = None    # exclusive

    def as_tuple(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        return self.start, self.end


@dataclass
class Selection:
    window: Optional[TimeWindow]
    mode: str                # 'INTELLIGENT' | 'HARD'
    confidence: float        # 0..1
    # [{'why': str, 'window': (start,end), 'confidence': float}]
    trace: List[Dict] = field(default_factory=list)

# ---------------------------
# Helpers
# ---------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _mk_day(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _month_end(y: int, m: int) -> datetime:
    # exclusive end = 1st of next month
    if m == 12:
        return _mk_day(y + 1, 1, 1)
    return _mk_day(y, m + 1, 1)


def _quarter_bounds_q1_jan(y: int, q: int) -> Tuple[datetime, datetime]:
    """
    Calendar quarters with Q1=Jan–Mar, Q2=Apr–Jun, Q3=Jul–Sep, Q4=Oct–Dec.
    Returns (start_inclusive, end_exclusive) in UTC.
    """
    if q == 1:  # Jan-Mar
        s = _mk_day(y, 1, 1)
        e = _mk_day(y, 4, 1)
    elif q == 2:  # Apr-Jun
        s = _mk_day(y, 4, 1)
        e = _mk_day(y, 7, 1)
    elif q == 3:  # Jul-Sep
        s = _mk_day(y, 7, 1)
        e = _mk_day(y, 10, 1)
    elif q == 4:  # Oct-Dec
        s = _mk_day(y, 10, 1)
        e = _mk_day(y, 1, 1).replace(year=y + 1)
    else:
        raise ValueError("quarter must be 1..4")
    return s, e


def _fy_bounds_april_start(fy_label: int) -> Tuple[datetime, datetime]:
    """
    Fiscal Year label == calendar year of its start (per your doc).
    FY25 → 2025-04-01 .. 2026-04-01 (exclusive).
    """
    s = _mk_day(fy_label, 4, 1)
    e = _mk_day(fy_label + 1, 4, 1)
    return s, e


def _q_in_fy_april_mapping(fy_label: int, q: int) -> Tuple[datetime, datetime]:
    """
    ⚠️ Per doc worked example: Fiscal start=April, yet Q2 == Apr–Jun of FY label.
    We mirror that behavior exactly:
       Q2 → Apr–Jun of FY(label)
       Q3 → Jul–Sep
       Q4 → Oct–Dec
       Q1 → Jan–Mar (of FY(label)+1)
    """
    if q == 2:
        return _mk_day(fy_label, 4, 1), _mk_day(fy_label, 7, 1)
    if q == 3:
        return _mk_day(fy_label, 7, 1), _mk_day(fy_label, 10, 1)
    if q == 4:
        return _mk_day(fy_label, 10, 1), _mk_day(fy_label + 1, 1, 1)
    if q == 1:
        return _mk_day(fy_label + 1, 1, 1), _mk_day(fy_label + 1, 4, 1)
    raise ValueError("quarter must be 1..4")


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# ---------------------------
# Parsers
# ---------------------------
_re_iso_date = re.compile(
    r"\b(20\d{2}|19\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
_re_asof = re.compile(r"\b(as\s*of|asof|on)\b", re.IGNORECASE)
_re_between = re.compile(
    r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_re_last_ndays = re.compile(r"\blast\s+(\d{1,3})\s+days?\b", re.IGNORECASE)
_re_yesterday = re.compile(r"\byesterday\b", re.IGNORECASE)
_re_today = re.compile(r"\b(today|now)\b", re.IGNORECASE)
_re_q_fy = re.compile(r"\bq([1-4])\s*fy\s*(\d{2,4})\b", re.IGNORECASE)


def parse_explicit_between(text: str) -> Optional[TimeWindow]:
    m = _re_between.search(text)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    A = datetime.fromisoformat(a).replace(tzinfo=timezone.utc)
    B = datetime.fromisoformat(b).replace(tzinfo=timezone.utc)
    # ensure end exclusive
    if B <= A:
        B = A
    return TimeWindow(A, B)


def parse_asof(text: str, now_utc: datetime) -> Optional[TimeWindow]:
    if _re_today.search(text) or _re_asof.search(text):
        # prefer explicit YYYY-MM-DD if present with "as of"
        m = _re_iso_date.search(text)
        if m:
            A = datetime.fromisoformat(m.group(0)).replace(tzinfo=timezone.utc)
            return TimeWindow(A, A)  # AS_OF day (degenerate window)
        # plain "as of today/now"
        A = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        return TimeWindow(A, A)
    return None


def parse_last_ndays(text: str, now_utc: datetime) -> Optional[TimeWindow]:
    m = _re_last_ndays.search(text)
    if not m:
        return None
    n = int(m.group(1))
    end = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + \
        timedelta(days=1)  # exclusive end = tomorrow 00:00
    start = end - timedelta(days=n)
    return TimeWindow(start, end)


def parse_yesterday(text: str, now_utc: datetime) -> Optional[TimeWindow]:
    if not _re_yesterday.search(text):
        return None
    end = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)
    return TimeWindow(start, end)


def parse_q_fy(text: str, fiscal_start_month: int = 4) -> Optional[TimeWindow]:
    """
    Parses patterns like 'Q2 FY25' or 'Q3 FY2026'.
    Follows your doc's April-start convention for Q mapping.
    """
    m = _re_q_fy.search(text)
    if not m:
        return None
    q = int(m.group(1))
    fy_raw = int(m.group(2))
    # Normalize FY label: '25' → 2025   (keeps doc convention)
    fy_label = fy_raw if fy_raw >= 100 else (2000 + fy_raw)
    if fiscal_start_month != 4:
        # For non-April starts (not used in Lite), fall back to calendar quarters
        s, e = _quarter_bounds_q1_jan(fy_label, q)
        return TimeWindow(s, e)
    s, e = _q_in_fy_april_mapping(fy_label, q)
    return TimeWindow(s, e)


# ---------------------------
# Router (INTELLIGENT vs HARD)
# ---------------------------
TAU_HARD = 0.80  # same spirit as playbook
TAU_LOW = 0.35


def normalize_query_time(
    query: str,
    now_utc: Optional[datetime] = None,
    fiscal_start_month: int = 4,
) -> Selection:
    """
    Returns a Selection(window, mode, confidence, trace)
    """
    now = now_utc or _now_utc()
    q = _strip(query)

    trace: List[Dict] = []
    candidates: List[Tuple[TimeWindow, float, str]] = []  # (window, conf, why)

    # HARD candidates (explicit)
    tw = parse_explicit_between(q)
    if tw:
        candidates.append((tw, 0.95, "explicit_between"))

    tw = parse_asof(q, now)
    if tw:
        candidates.append((tw, 0.92, "as_of"))

    tw = parse_q_fy(q, fiscal_start_month=fiscal_start_month)
    if tw:
        candidates.append((tw, 0.90, "q_fy"))

    # INTELLIGENT candidates (soft)
    tw = parse_last_ndays(q, now)
    if tw:
        candidates.append((tw, 0.70, "last_ndays"))

    tw = parse_yesterday(q, now)
    if tw:
        candidates.append((tw, 0.60, "yesterday"))

    # Fallback soft window: last 180 days (kept modest; real density check happens upstream)
    if not candidates:
        end = now.replace(hour=0, minute=0, second=0,
                          microsecond=0) + timedelta(days=1)
        start = end - timedelta(days=180)
        candidates.append((TimeWindow(start, end), 0.40, "fallback_last_180d"))

    # Pick top by confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_tw, best_conf, reason = candidates[0]
    for twi, ci, why in candidates:
        s = twi.start.isoformat() if twi.start else None
        e = twi.end.isoformat() if twi.end else None
        trace.append({"why": why, "confidence": ci, "start": s, "end": e})

    mode = "HARD" if best_conf >= TAU_HARD else "INTELLIGENT"
    return Selection(window=best_tw, mode=mode, confidence=best_conf, trace=trace)

# ---------------------------
# CLI for quick checks
# ---------------------------


def _fmt(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else "None"


def main():
    ap = argparse.ArgumentParser(
        description="ChronoRAG-Lite temporal normalizer")
    ap.add_argument("--q", "--query", dest="query",
                    required=True, help="natural language query")
    ap.add_argument("--fy_start", type=int, default=4,
                    help="fiscal start month (default 4=April)")
    args = ap.parse_args()

    sel = normalize_query_time(args.query, fiscal_start_month=args.fy_start)
    print(f"MODE={sel.mode}  CONF={sel.confidence:.2f}")
    if sel.window:
        print(f"WINDOW: [{_fmt(sel.window.start)} .. {_fmt(sel.window.end)})")
    print("TRACE:")
    for t in sel.trace:
        print(
            f"  - {t['why']:>18s}  conf={t['confidence']:.2f}  {t['start']} .. {t['end']}")


if __name__ == "__main__":
    main()
