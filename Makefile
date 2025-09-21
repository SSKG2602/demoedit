ENV_NAME=chronorag-lite

.PHONY: env api ui ingest index ask test
env:
	conda env create -f envs/chronorag-lite.yaml || conda env update -f envs/chronorag-lite.yaml --prune
	@echo "Activate: conda activate $(ENV_NAME)"

api:
	uvicorn app.api:app --reload --port 8000

ui:
	streamlit run ui/app.py

ingest:
	python -m ingest.loader --path ./ingest/demo_dataset --chunk_size 400 --overlap 40

index:
	python -m ingest.build_faiss --prefer_fastembed --model BAAI/bge-small-en-v1.5 --batch 64

ask:
	python -m tools.ask_cli "Who is the CEO as of 2025-08-15?"

test:
	pytest -q
