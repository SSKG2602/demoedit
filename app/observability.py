#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Observability
Goals (Lite, spec-aligned):
  • Minimal counters, timings, and event logs with near-zero deps
  • Optional OpenTelemetry spans if opentelemetry is installed
  • Debug-friendly printing for local dev on macOS+conda

Usage:
  from app.observability import obs, span, log_event, incr

  with span("orchestrator.retrieve"):
      ... work ...
      incr("retrieval.ok")

Design:
  • No global thread contention; simple per-process counters.
  • OTEL is best-effort: if the package/env isn’t present, we just no-op the exporter.
"""

from __future__ import annotations
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

# ---------------------------
# Simple in-process metrics
# ---------------------------


@dataclass
class Metrics:
    counters: Dict[str, int] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)

    def incr(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def observe_ms(self, key: str, ms: float) -> None:
        # last-value wins; callers can use distinct keys per section
        self.timings_ms[key] = ms


@dataclass
class Observer:
    metrics: Metrics = field(default_factory=Metrics)
    verbose: bool = True

    # OTEL lazy init
    _otel_ready: bool = False
    _otel_tracer = None

    def _try_init_otel(self) -> None:
        if self._otel_ready:
            return
        try:
            from opentelemetry import trace  # type: ignore
            self._otel_tracer = trace.get_tracer("chronorag-lite")
            self._otel_ready = True
            if self.verbose:
                print("[obs] OpenTelemetry available")
        except Exception:
            self._otel_ready = False
            self._otel_tracer = None
            if self.verbose:
                print("[obs] OpenTelemetry not installed; continuing without spans")

    def log(self, msg: str, **kv) -> None:
        if self.verbose:
            if kv:
                print(
                    f"[obs] {msg} :: {json.dumps(kv, ensure_ascii=False)}", flush=True)
            else:
                print(f"[obs] {msg}", flush=True)

    def incr(self, key: str, n: int = 1) -> None:
        self.metrics.incr(key, n)

    def observe_ms(self, key: str, ms: float) -> None:
        self.metrics.observe_ms(key, ms)

    @contextmanager
    def span(self, name: str, **attrs):
        """
        Context manager that times a block and (optionally) emits an OTEL span.
        """
        t0 = time.perf_counter()
        self._try_init_otel()
        span_ctx = None
        if self._otel_ready and self._otel_tracer:
            try:
                span_ctx = self._otel_tracer.start_as_current_span(name)
                span = span_ctx.__enter__()
                for k, v in attrs.items():
                    try:
                        span.set_attribute(str(k), v)
                    except Exception:
                        pass
            except Exception:
                span_ctx = None

        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.observe_ms(name, ms)
            self.log(f"span_end:{name}", ms=round(ms, 2), **attrs)
            if span_ctx is not None:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception:
                    pass


# Singleton
obs = Observer()

# ---------------------------------
# Convenience top-level functions
# ---------------------------------


def incr(key: str, n: int = 1) -> None:
    obs.incr(key, n)


def observe_ms(key: str, ms: float) -> None:
    obs.observe_ms(key, ms)


def log_event(event: str, **kv) -> None:
    obs.log(event, **kv)

# Alias for context manager


def span(name: str, **attrs):
    return obs.span(name, **attrs)
