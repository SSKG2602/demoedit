#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — Central Settings
Single source of truth for paths, model names, ports, and policy weights.
Uses environment variables with safe defaults for local dev.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import os

# Optional dependencies: support both Pydantic v1 and v2, and run without pydantic-settings
_IMPL = "unknown"
model_validator = None  # type: ignore[assignment]
root_validator = None  # type: ignore[assignment]
try:
    # Pydantic v2 preferred path
    from pydantic import Field, model_validator  # type: ignore
    from pydantic_settings import BaseSettings as _BaseSettings, SettingsConfigDict  # type: ignore
    _IMPL = "pydantic-v2"
except Exception:  # pydantic-settings missing or older pydantic
    try:
        import pydantic as _p
        from pydantic import Field  # type: ignore
        if getattr(_p, "__version__", "2").startswith("1."):
            from pydantic import BaseSettings as _BaseSettings, root_validator  # type: ignore
            SettingsConfigDict = None  # type: ignore
            _IMPL = "pydantic-v1"
        else:
            # Pydantic v2 without pydantic-settings → use shim (no Field needed here)
            _IMPL = "shim"
            Field = None  # type: ignore
            _BaseSettings = object  # type: ignore
            SettingsConfigDict = None  # type: ignore
    except Exception:
        # No pydantic at all → shim
        _IMPL = "shim"
        Field = None  # type: ignore
        _BaseSettings = object  # type: ignore
        SettingsConfigDict = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


_ENV_ALIAS_MAP = {
    "gamma_auth": "gamma_authority",
}


def _normalize_env_values(raw: Any) -> Any:
    """Map legacy CR_* keys to model fields for BaseSettings."""

    if not isinstance(raw, dict):
        return raw

    normalized: Dict[str, Any] = {}
    for key, value in raw.items():
        key_str = str(key)
        key_lower = key_str.lower()

        if key_lower.startswith("cr_"):
            key_lower = key_lower[3:]

        field_name = _ENV_ALIAS_MAP.get(key_lower, key_lower)
        normalized.setdefault(field_name, value)

    return normalized


if _IMPL in {"pydantic-v1", "pydantic-v2"}:
    # Pydantic-backed settings (v1 or v2)
    class Settings(_BaseSettings):
        # ---- service ----
        api_host: str = Field("0.0.0.0")  # type: ignore
        api_port: int = Field(8000)  # type: ignore
        log_verbose: bool = Field(True)  # type: ignore

        # ---- data & index ----
        data_dir: Path = Field(ROOT / "data")  # type: ignore[arg-type]
        index_dir: Path = Field(ROOT / "indexes")  # type: ignore[arg-type]
        chunks_csv: Path = Field(ROOT / "data" / "chunks.csv")  # type: ignore[arg-type]
        meta_parquet: Path = Field(
            ROOT / "data" / "chunk_meta.parquet"
        )  # type: ignore[arg-type]
        ids_npy: Path = Field(ROOT / "data" / "chunk_ids.npy")  # type: ignore[arg-type]
        faiss_index: Path = Field(
            ROOT / "indexes" / "faiss.index"
        )  # type: ignore[arg-type]

        # ---- models ----
        embed_model: str = Field("BAAI/bge-small-en-v1.5")  # type: ignore
        prefer_fastembed: bool = Field(True)  # type: ignore
        llm_model: str = Field("openai/gpt-oss-20b")  # type: ignore
        llm_max_new_tokens: int = Field(256)  # type: ignore

        # ---- retrieval weights (POLAR-lite) ----
        alpha: float = Field(0.55)  # type: ignore
        beta_time: float = Field(0.25)  # type: ignore
        gamma_authority: float = Field(0.15)  # type: ignore
        delta_age: float = Field(0.05)  # type: ignore

        # ---- controller (DHQC-lite) ----
        tau: float = Field(0.80)  # type: ignore
        delta: float = Field(0.20)  # type: ignore
        n_max: int = Field(3)  # type: ignore

        if _IMPL == "pydantic-v2":
            model_config = SettingsConfigDict(  # type: ignore[call-arg]
                env_file=str(ROOT / ".env"),
                env_file_encoding="utf-8",
                case_sensitive=False,
                extra="ignore",
            )

            @model_validator(mode="before")  # type: ignore[misc]
            @classmethod
            def _coerce_env(cls, data: Any) -> Any:
                return _normalize_env_values(data)

        else:
            class Config:  # type: ignore[no-redef]
                env_file = str(ROOT / ".env")
                env_file_encoding = "utf-8"
                case_sensitive = False
                extra = "ignore"

            @root_validator(pre=True)  # type: ignore[misc]
            def _coerce_env(cls, values: Any) -> Any:  # type: ignore[override]
                return _normalize_env_values(values)

    settings = Settings()
else:
    # Shim implementation without pydantic-settings (works with Pydantic v2-only envs)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=str(ROOT / ".env"))
    except Exception:
        pass

    def _get_env(name: str, default, cast):
        val = os.getenv(name)
        if val is None:
            return default
        try:
            if cast is bool:
                return str(val).strip().lower() in {"1", "true", "yes", "on"}
            if cast is int:
                return int(val)
            if cast is float:
                return float(val)
            if cast is Path:
                return Path(val)
            return cast(val)
        except Exception:
            return default

    class Settings:
        # ---- service ----
        api_host: str
        api_port: int
        log_verbose: bool

        # ---- data & index ----
        data_dir: Path
        index_dir: Path
        chunks_csv: Path
        meta_parquet: Path
        ids_npy: Path
        faiss_index: Path

        # ---- models ----
        embed_model: str
        prefer_fastembed: bool
        llm_model: str
        llm_max_new_tokens: int

        # ---- retrieval weights (POLAR-lite) ----
        alpha: float
        beta_time: float
        gamma_authority: float
        delta_age: float

        # ---- controller (DHQC-lite) ----
        tau: float
        delta: float
        n_max: int

        def __init__(self) -> None:
            # service
            self.api_host = _get_env("CR_API_HOST", "0.0.0.0", str)
            self.api_port = _get_env("CR_API_PORT", 8000, int)
            self.log_verbose = _get_env("CR_LOG_VERBOSE", True, bool)

            # data & index
            self.data_dir = ROOT / "data"
            self.index_dir = ROOT / "indexes"
            self.chunks_csv = ROOT / "data" / "chunks.csv"
            self.meta_parquet = ROOT / "data" / "chunk_meta.parquet"
            self.ids_npy = ROOT / "data" / "chunk_ids.npy"
            self.faiss_index = ROOT / "indexes" / "faiss.index"

            # models
            self.embed_model = _get_env("CR_EMBED_MODEL", "BAAI/bge-small-en-v1.5", str)
            self.prefer_fastembed = _get_env("CR_PREFER_FASTEMBED", True, bool)
            self.llm_model = _get_env("CR_LLM_MODEL", "openai/gpt-oss-20b", str)
            self.llm_max_new_tokens = _get_env("CR_LLM_MAX_NEW_TOKENS", 256, int)

            # retrieval weights
            self.alpha = _get_env("CR_ALPHA", 0.55, float)
            self.beta_time = _get_env("CR_BETA_TIME", 0.25, float)
            self.gamma_authority = _get_env("CR_GAMMA_AUTH", 0.15, float)
            self.delta_age = _get_env("CR_DELTA_AGE", 0.05, float)

            # controller
            self.tau = _get_env("CR_TAU", 0.80, float)
            self.delta = _get_env("CR_DELTA", 0.20, float)
            self.n_max = _get_env("CR_N_MAX", 3, int)

    settings = Settings()
