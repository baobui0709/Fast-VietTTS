#!/usr/bin/env python3
"""Benchmark optimization variants for Fast-VietTTS."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import FastVietTTS
from src.speed_utils import compile_submodules, cuda_cleanup, enable_tf32, warmup_engine


TEXTS = {
    "short": "Xin chào, đây là bài kiểm tra ngắn cho Fast-VietTTS.",
    "medium": (
        "Tiêu Viêm đứng trước cổng học viện, ánh mắt nhìn về phía xa, "
        "lòng tràn đầy quyết tâm bước vào con đường tu luyện mới."
    ),
    "long": (
        "Việt Nam là một quốc gia nằm ở Đông Nam Á với lịch sử lâu đời, văn hóa phong phú "
        "và con người thân thiện. Công nghệ trí tuệ nhân tạo đang mở ra nhiều cơ hội mới "
        "cho việc tạo sách nói và trợ lý giọng nói tự nhiên."
    ),
}


def bench_once(engine: FastVietTTS, text: str, ref: str | None = None) -> dict:
    """Run one timed generation."""
    cuda_cleanup()
    t0 = time.perf_counter()
    wav = engine.generate(text=text, language="vi", audio_prompt=ref)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    audio_sec = wav.shape[-1] / 24000
    return {
        "chars": len(text),
        "audio_sec": audio_sec,
        "elapsed_sec": elapsed,
        "rtf": elapsed / max(audio_sec, 1e-6),
    }


def run_variant(name: str, ref: str | None, compile_targets: tuple[str, ...]) -> list[dict]:
    """Run one optimization variant."""
    enable_tf32()
    engine = FastVietTTS(device="cuda" if torch.cuda.is_available() else "cpu", use_fp16=False, compile_model=False)
    compiled = compile_submodules(engine, compile_targets) if compile_targets else []
    warmup_sec = warmup_engine(engine)

    rows = []
    for case, text in TEXTS.items():
        result = bench_once(engine, text, ref=ref)
        result.update({"variant": name, "case": case, "compiled": ",".join(compiled), "warmup_sec": warmup_sec})
        rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast-VietTTS speed benchmark")
    parser.add_argument("--ref", default=None, help="Optional reference wav")
    parser.add_argument("--output", default="benchmark_outputs/speed_results.json")
    parser.add_argument("--try-compile", action="store_true", help="Also test torch.compile variants")
    args = parser.parse_args()

    variants: list[tuple[str, tuple[str, ...]]] = [("tf32_warmup", tuple())]
    if args.try_compile:
        variants += [
            ("compile_t3", ("t3",)),
            ("compile_s3gen", ("s3gen",)),
            ("compile_t3_s3gen", ("t3", "s3gen")),
        ]

    all_rows: list[dict] = []
    for name, targets in variants:
        print(f"Running variant: {name}")
        try:
            all_rows.extend(run_variant(name, args.ref, targets))
        except Exception as exc:
            all_rows.append({"variant": name, "error": repr(exc)})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Speed Benchmark — Agent 10",
        "",
        "| Variant | Case | Chars | Audio sec | Elapsed sec | RTF | Compiled |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in all_rows:
        if "error" in row:
            md_lines.append(f"| {row['variant']} | ERROR | | | | | {row['error']} |")
            continue
        md_lines.append(
            f"| {row['variant']} | {row['case']} | {row['chars']} | "
            f"{row['audio_sec']:.2f} | {row['elapsed_sec']:.2f} | {row['rtf']:.2f} | {row.get('compiled','')} |"
        )

    Path("docs/SPEED_RESULTS.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("✅ Saved", out)
    print("✅ Saved docs/SPEED_RESULTS.md")


if __name__ == "__main__":
    main()
