#!/usr/bin/env python3
"""Benchmark Agent12 reference conditioning cache."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import FastVietTTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark reference conditioning cache")
    parser.add_argument("--ref", default="wavs/00c3bca9-4885-4b2e-8139-10e09df6143c.wav")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--text", default="Tôi có 3 người bạn. Năm 2025 tôi đến TP.HCM.")
    parser.add_argument("--output", default="docs/REF_CACHE_RESULTS.md")
    args = parser.parse_args()

    engine = FastVietTTS(device="cuda" if torch.cuda.is_available() else "cpu", use_fp16=False, compile_model=False)
    rows = []

    for i in range(1, args.repeat + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        audio = engine.generate(text=args.text, language="vi", audio_prompt=args.ref)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        audio_sec = audio.shape[-1] / 24000
        rows.append({"run": i, "audio_sec": audio_sec, "elapsed_sec": elapsed, "rtf": elapsed / max(audio_sec, 1e-6)})

    lines = [
        "# Reference Cache Results — Agent 12",
        "",
        "| Run | Audio sec | Elapsed sec | RTF |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['run']} | {row['audio_sec']:.2f} | {row['elapsed_sec']:.2f} | {row['rtf']:.2f} |")

    if len(rows) >= 2:
        first = rows[0]["elapsed_sec"]
        later_avg = sum(r["elapsed_sec"] for r in rows[1:]) / (len(rows) - 1)
        lines += ["", f"First run: **{first:.2f}s**", f"Later avg: **{later_avg:.2f}s**", f"Speedup after cache: **{first / max(later_avg, 1e-6):.2f}x**"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(Path(args.output).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
