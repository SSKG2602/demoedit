#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Central Settings
Single source of truth for paths, model names, ports, and policy weights.
Uses environment variables with safe defaults for local dev.
"""
from __future__ import annotations
from pathlib import Path
from pydantic import BaseSettings, Field

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # ---- service ----
    api_host: str = Field("0.0.0.0", env="CR_API_HOST")
    api_port: int = Field(8000, env="CR_API_PORT")
    log_verbose: bool = Field(True, env="CR_LOG_VERBOSE")

    # ---- data & index ----
    data_dir: Path = Field(default=ROOT / "data")
    index_dir: Path = Field(default=ROOT / "indexes")
    chunks_csv: Path = Field(default=ROOT / "data" / "chunks.csv")
    meta_parquet: Path = Field(default=ROOT / "data" / "chunk_meta.parquet")
    ids_npy: Path = Field(default=ROOT / "data" / "chunk_ids.npy")
    faiss_index: Path = Field(default=ROOT / "indexes" / "faiss.index")

    # ---- models ----
    embed_model: str = Field("BAAI/bge-small-en-v1.5", env="CR_EMBED_MODEL")
    prefer_fastembed: bool = Field(True, env="CR_PREFER_FASTEMBED")
    llm_model: str = Field("openai/gpt-oss-20b", env="CR_LLM_MODEL")
    llm_max_new_tokens: int = Field(256, env="CR_LLM_MAX_NEW_TOKENS")

    # ---- retrieval weights (POLAR-lite) ----
    alpha: float = Field(0.55, env="CR_ALPHA")
    beta_time: float = Field(0.25, env="CR_BETA_TIME")
    gamma_authority: float = Field(0.15, env="CR_GAMMA_AUTH")
    delta_age: float = Field(0.05, env="CR_DELTA_AGE")

    # ---- controller (DHQC-lite) ----
    tau: float = Field(0.80, env="CR_TAU")
    delta: float = Field(0.20, env="CR_DELTA")
    n_max: int = Field(3, env="CR_N_MAX")

    class Config:
        env_file = str(ROOT / ".env")
        env_file_encoding = "utf-8"


settings = Settings()
