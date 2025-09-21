# ⏱️ ChronoRAG-Lite

A minimal, **time-aware** RAG pipeline with:

- Temporal normalization (INTELLIGENT/HARD)
- Hybrid retrieval (cosine ANN + time weights)
- DHQC-Lite controller (early-exit planning)
- Answer synthesis (local HF LLM or deterministic fallback)
- Attribution Cards (auditable evidence receipts)

> **Goal:** Ship a lean public PoC (15–20% of the full system) without revealing proprietary controllers/graphs.

## Quickstart

```bash
# 0) env (Mac + conda)
conda activate chronorag-lite

# 1) ingest some text
python -m ingest.loader --path ./ingest/demo_dataset --chunk_size 400 --overlap 40

# 2) build index
python -m ingest.build_faiss --prefer_fastembed --model BAAI/bge-small-en-v1.5 --batch 64

# 3) run API
uvicorn app.api:app --reload --port 8000
```
