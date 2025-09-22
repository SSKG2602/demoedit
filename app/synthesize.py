#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Answer Synthesizer
Spec alignment:
  • Role prompt reflects time badges (MODE, WINDOW) and citation discipline.
  • Evidence presented as ranked snippets with windows (Lite approximations).
  • Uses LLMProvider if available; else returns deterministic fallback.

Inputs match orchestrator contract.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from app.llm_provider import LLMProvider, LLMConfig
try:
    # Prefer centralized settings for model defaults
    from app.settings import settings
    _DEFAULT_LLM_MODEL = settings.llm_model
    _DEFAULT_MAX_NEW_TOKENS = settings.llm_max_new_tokens
except Exception:
    # Fallback to hardcoded safe defaults
    _DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    _DEFAULT_MAX_NEW_TOKENS = 256

# ---------------------------
# Public API
# ---------------------------


def generate_answer(
    query: str,
    candidates: List[Any],  # list[Candidate] from retriever
    # {"mode":..,"confidence":..,"start":..,"end":..}
    time_filters: Dict[str, Any],
    mode: str,
    llm_model: str = _DEFAULT_LLM_MODEL,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """
    Returns a short answer followed by 2–3 concise rationale bullets.
    """
    # Build instruction with time badges (per Playbook)
    window = f"[{time_filters.get('start')} .. {time_filters.get('end')})"
    sys_prompt = (
        "You are ChronoRAG Answerer.\n"
        "Use ONLY the provided evidence. Respect the time filters.\n"
        "If evidence conflicts across disjoint windows, prefer the one aligned with the active window.\n"
        "Cite the window in-line (e.g., [WINDOW]). Keep the answer concise."
    )

    # Prepare evidence list (ranked, already fused)
    ev_lines = []
    for i, c in enumerate(candidates[:8], 1):
        w_from = getattr(c, "valid_from", None) or getattr(
            c, "published_at", None)
        w_to = getattr(c, "valid_to", None) or w_from
        w = f"[{w_from} .. {w_to})" if (w_from or w_to) else "[unknown]"
        src = getattr(c, "source_path", None) or getattr(c, "doc_id", "")
        quote = (getattr(c, "text", "") or "").replace("\n", " ")
        ev_lines.append(f"- ({i:02d}) {w} :: {quote[:400]} — src={src}")

    user_prompt = (
        f"[MODE] {mode} • [WINDOW] {window}\n"
        f"Question: {query}\n\n"
        "Evidence (ranked):\n" + "\n".join(ev_lines) + "\n\n"
        "Respond in two parts:\n"
        "1) A one-paragraph answer grounded in evidence within [WINDOW].\n"
        "2) Then 2–3 bullets explaining the reasoning and pointing to the most relevant evidence indices.\n"
    )

    provider = LLMProvider(
        LLMConfig(model_name=llm_model, max_new_tokens=max_new_tokens))
    if provider.is_available():
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return provider.generate(messages, max_new_tokens=max_new_tokens)
        except Exception:
            pass  # fall through to fallback

    # Deterministic fallback (same shape as orchestrator fallback, but with time badges)
    head = f"Answer (Lite): mode={mode}, window={window}"
    bullets = []
    for i, c in enumerate(candidates[:3], 1):
        src = getattr(c, "source_path", None) or getattr(c, "doc_id", "")
        bullets.append(
            f"- {getattr(c, 'text', '')[:180].replace(chr(10),' ')} (src={src})")
    return head + "\n" + "\n".join(bullets)
