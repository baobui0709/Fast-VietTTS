"""Speed utilities for Fast-VietTTS/ViterBox fallback."""

from __future__ import annotations

import gc
import time
from typing import Any

import torch


def enable_tf32() -> None:
    """Enable TF32 matmul/cudnn on NVIDIA GPUs when available."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def cuda_cleanup() -> None:
    """Release Python and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def warmup_engine(engine: Any, text: str = "Xin chào.") -> float:
    """Run a short warmup generation and return elapsed seconds."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = engine.generate(text, language="vi")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def compile_submodules(engine: Any, targets: tuple[str, ...] = ("t3", "s3gen")) -> list[str]:
    """Try torch.compile on selected ViterBox submodules."""
    compiled: list[str] = []
    if not hasattr(torch, "compile"):
        return compiled

    model = getattr(engine, "model", engine)
    for name in targets:
        module = getattr(model, name, None)
        if module is None or not hasattr(module, "forward"):
            continue
        try:
            setattr(model, name, torch.compile(module, mode="reduce-overhead"))
            compiled.append(name)
        except Exception:
            continue
    return compiled
