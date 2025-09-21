#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — E2E Tests
Runs against the FastAPI app in-process (no server needed).
Assumes data/index artifacts exist.
"""
from __future__ import annotations
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_retrieve_basic():
    body = {"query": "Revenue in Q2 FY25", "k": 5, "time_mode": "HARD"}
    r = client.post("/api/v1/retrieve", json=body)
    assert r.status_code == 200
    js = r.json()
    assert "candidates" in js and isinstance(js["candidates"], list)
    assert "time_filters" in js
    assert js["time_filters"]["mode"] in ("HARD", "INTELLIGENT")


def test_answer_basic():
    body = {"query": "Who is the CEO as of 2025-08-15?", "k": 4}
    r = client.post("/api/v1/answer", json=body)
    assert r.status_code == 200
    js = r.json()
    assert "answer" in js and isinstance(js["answer"], str)
    assert "controller_stats" in js
