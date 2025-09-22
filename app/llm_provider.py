#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChronoRAG-Lite — LLM Provider
Backends:
  • HF Transformers (default) using model name (e.g., Qwen/Qwen2.5-3B-Instruct)
  • llama.cpp (auto when CR_LLM_BACKEND=llama.cpp or CR_LLM_PATH endswith .gguf)

Environment (see app/settings.py / .env):
  CR_LLM_MODEL   — HF model name (e.g., Qwen/Qwen2.5-3B-Instruct)
  CR_LLM_BACKEND — "hf" | "llama.cpp" (optional; auto if LLM_PATH is .gguf)
  CR_LLM_PATH    — local path to .gguf for llama.cpp
  CR_LLM_MAX_NEW_TOKENS, CR_LLM_TEMPERATURE, CR_LLM_TOP_P
"""

from __future__ import annotations
import os
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional

# Resolve defaults from centralized settings if available; otherwise env
try:  # Prefer app.settings to honor .env
    from app.settings import settings  # type: ignore
    _DEFAULT_LLM_MODEL = settings.llm_model
    _DEFAULT_MAX_NEW_TOKENS = settings.llm_max_new_tokens
except Exception:
    _DEFAULT_LLM_MODEL = os.environ.get(
        "CR_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    try:
        _DEFAULT_MAX_NEW_TOKENS = int(
            os.environ.get("CR_LLM_MAX_NEW_TOKENS", "256"))
    except Exception:
        _DEFAULT_MAX_NEW_TOKENS = 256


@dataclass
class LLMConfig:
    model_name: str = _DEFAULT_LLM_MODEL  # HF name
    # "hf" | "llama.cpp"
    backend: str = os.environ.get("CR_LLM_BACKEND", "hf")
    gguf_path: Optional[str] = os.environ.get(
        "CR_LLM_PATH")                 # ./models/model.gguf
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
    temperature: float = float(os.environ.get("CR_LLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.environ.get("CR_LLM_TOP_P", "0.95"))


class LLMProvider:
    def __init__(self, cfg: Optional[LLMConfig] = None) -> None:
        self.cfg = cfg or LLMConfig()
        # auto backend if .gguf provided
        if self.cfg.gguf_path and self.cfg.gguf_path.endswith(".gguf"):
            self.cfg.backend = "llama.cpp"
        self._tok = None
        self._model = None
        self._failed_reason: Optional[str] = None
        self._loaded = False

    def _log(self, msg: str) -> None:
        print(f"[llm] {msg}", flush=True)

    # ----------------- Public API -----------------
    def is_available(self) -> bool:
        self._lazy_load()
        return self._model is not None

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
        self._lazy_load()
        if self._model is None:
            return self._fallback(messages)
        if self.cfg.backend == "llama.cpp":
            return self._gen_llama(messages, max_new_tokens)
        return self._gen_hf(messages, max_new_tokens)

    # ----------------- Backends -------------------
    def _lazy_load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if self.cfg.backend == "llama.cpp":
                self._load_llama()
            else:
                self._load_hf()
        except Exception as e:
            self._failed_reason = f"{type(e).__name__}: {e}"
            self._log("FAILED to load LLM; using fallback")
            traceback.print_exc()

    def _select_dtype(self, torch_mod, device):
        """Pick the safest tensor dtype for the current accelerator."""
        if device.type == "cpu":
            return torch_mod.float32
        if device.type == "cuda":
            supports_bf16 = False
            if hasattr(torch_mod, "cuda") and hasattr(torch_mod.cuda, "is_bf16_supported"):
                try:
                    supports_bf16 = bool(torch_mod.cuda.is_bf16_supported())
                except Exception:
                    supports_bf16 = False
            if supports_bf16 and hasattr(torch_mod, "bfloat16"):
                return torch_mod.bfloat16
            if supports_bf16:
                self._log("CUDA device reports bfloat16 support but torch lacks dtype; falling back to float16")
            else:
                self._log("CUDA device does not support bfloat16; using float16")
            return getattr(torch_mod, "float16", torch_mod.float32)
        if device.type == "mps":
            return getattr(torch_mod, "float16", torch_mod.float32)
        return torch_mod.float32

    # HF Transformers path (with Qwen-friendly options)
    def _load_hf(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        name = self.cfg.model_name
        trust_remote = name.startswith(
            "Qwen/") or bool(int(os.environ.get("CR_TRUST_REMOTE_CODE", "1")))

        # Determine compute device and dtype without requiring accelerate
        has_mps = hasattr(
            torch.backends, "mps") and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
        device = (
            torch.device("cuda") if has_cuda else
            torch.device("mps") if has_mps else
            torch.device("cpu")
        )
        dtype = self._select_dtype(torch, device)

        # Only use device_map if accelerate is available (to avoid the error you saw)
        try:
            import accelerate  # noqa: F401
            can_use_device_map = os.environ.get("CR_USE_DEVICE_MAP", "1").lower() not in {
                "0", "false", "no"}
        except Exception:
            can_use_device_map = False

        self._log(
            f"loading HF model: {name} (trust_remote_code={trust_remote}, device={device}, dtype={dtype}, device_map={'auto' if can_use_device_map else 'none'})")

        # Tokenizer
        self._tok = AutoTokenizer.from_pretrained(
            name, trust_remote_code=trust_remote)

        # Model load
        kwargs = dict(trust_remote_code=trust_remote, torch_dtype=dtype)
        if can_use_device_map:
            kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

        # If we didn't use device_map, move to the chosen device explicitly (for small models)
        if not can_use_device_map and device.type != "cpu":
            try:
                self._model.to(device=device, dtype=dtype)
            except Exception:
                # If moving fails (e.g., not enough memory), keep on CPU; generation may be slow
                self._log("Could not move model to GPU/MPS, keeping on CPU")

        # Make sure pad_token_id is set (some instruct models leave it None)
        if getattr(self._tok, "pad_token_id", None) is None and getattr(self._tok, "eos_token_id", None) is not None:
            self._tok.pad_token_id = self._tok.eos_token_id

        if has_mps:
            self._log("MPS available")

        self._log("HF model loaded")

    def _gen_hf(self, messages: List[Dict[str, str]], max_new_tokens: Optional[int]) -> str:
        from transformers import GenerationConfig
        import torch

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
            do_sample=(self.cfg.temperature > 0.0),
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
        )

        # Use tokenizer chat template (Qwen provides one)
        inputs = self._tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(**inputs, generation_config=gen_cfg)

        start = inputs["input_ids"].shape[-1]
        return self._tok.decode(out[0][start:], skip_special_tokens=True).strip()

    # llama.cpp (GGUF) path (kept for future low-RAM switch)
    def _load_llama(self) -> None:
        from llama_cpp import Llama  # type: ignore
        gguf = self.cfg.gguf_path
        if not gguf:
            raise RuntimeError(
                "CR_LLM_BACKEND=llama.cpp but CR_LLM_PATH is not set to a .gguf file")
        n_gpu_layers = 1  # Metal offload a bit; set 0 for pure CPU
        self._log(f"loading GGUF via llama.cpp: {gguf}")
        self._model = Llama(
            model_path=gguf,
            n_ctx=8192,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._tok = None
        self._log("llama.cpp model loaded")

    def _gen_llama(self, messages: List[Dict[str, str]], max_new_tokens: Optional[int]) -> str:
        sys_msg = ""
        for m in messages:
            if m["role"] == "system":
                sys_msg = m["content"].strip()
                break
        convo = []
        if sys_msg:
            convo.append(f"<<SYS>>\n{sys_msg}\n<</SYS>>\n")
        for m in messages:
            if m["role"] == "user":
                convo.append(f"USER: {m['content'].strip()}\n")
            elif m["role"] == "assistant":
                convo.append(f"ASSISTANT: {m['content'].strip()}\n")
        convo.append("ASSISTANT:")
        prompt = "".join(convo)

        out = self._model(
            prompt,
            max_tokens=max_new_tokens or self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            stop=["USER:", "ASSISTANT:"],
        )
        return (out["choices"][0]["text"] or "").strip()

    # ----------------- Fallback -------------------
    def _fallback(self, messages: List[Dict[str, str]]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = (m.get("content") or "").strip()
                break
        note = "LLM unavailable locally"
        if self._failed_reason:
            note += f" ({self._failed_reason})"
        return f"{note}. Evidence-based fallback will be used. Q: {last_user}"
