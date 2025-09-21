#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Streamlit UI
Simple panel to query the API, visualize answer + chips.
"""
from __future__ import annotations
import os
import requests
import streamlit as st

API = os.environ.get("CR_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="ChronoRAG-Lite", layout="wide")
st.title("⏱️ ChronoRAG-Lite")

with st.sidebar:
    st.header("Settings")
    time_mode = st.selectbox(
        "Time Mode", ["AUTO (router)", "INTELLIGENT", "HARD"], index=0)
    k = st.slider("Top-k", min_value=3, max_value=12, value=6, step=1)
    st.caption(f"API: {API}")
    st.divider()
    st.caption("Run API first: `uvicorn app.api:app --reload --port 8000`")

query = st.text_input(
    "Ask a question", value="Who is the CEO as of 2025-08-15?")
go = st.button("Ask")


def call_api(route, payload):
    url = f"{API}{route}"
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


if go and query.strip():
    tm = None if time_mode.startswith("AUTO") else time_mode
    payload = {"query": query, "k": k, "time_mode": tm}

    with st.spinner("Thinking…"):
        data = call_api("/api/v1/answer", payload)

    st.subheader("Answer")
    st.write(data["answer"])

    tf = data.get("time_filters", {})
    st.subheader("Time Filters")
    st.json(tf, expanded=False)

    card = data.get("attribution_card") or {}
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Badges")
        st.table(card.get("badges", []))
    with col2:
        st.subheader("Top Receipts")
        recs = card.get("receipts", [])[:8]
        for r in recs:
            st.markdown(
                f"**{r.get('rank', '?')}. {r.get('window','[unknown]')}**")
            st.caption(f"src: {r.get('source')}")
            st.write(r.get("quote", "")[:500])
            st.caption(f"scores: {r.get('scores')}")
            st.divider()
