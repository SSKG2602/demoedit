#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Ask CLI
Tiny client to hit the FastAPI endpoints from terminal.
"""
from __future__ import annotations
import argparse
import json
import sys
import requests


def pretty(obj):  # stable compact print
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ask ChronoRAG-Lite over HTTP")
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--time_mode", default=None,
                    choices=[None, "INTELLIGENT", "HARD"])
    ap.add_argument("--action", default="answer",
                    choices=["retrieve", "answer"])
    ap.add_argument("query", help="natural language question")
    args = ap.parse_args(argv)

    url = f"{args.host}/api/v1/{'retrieve' if args.action=='retrieve' else 'answer'}"
    body = {"query": args.query, "k": args.k, "time_mode": args.time_mode}
    r = requests.post(url, json=body, timeout=120)
    r.raise_for_status()
    out = r.json()

    if args.action == "answer":
        print("\n--- ANSWER ---\n")
        print(out["answer"])
        print("\n--- BADGES ---\n")
        pretty(out.get("attribution_card", {}).get("badges", []))
        print("\n--- TOP RECEIPTS ---\n")
        pretty(out.get("attribution_card", {}).get("receipts", [])[:3])
    else:
        print("\n--- RETRIEVAL ---\n")
        pretty(out)


if __name__ == "__main__":
    sys.exit(main())
