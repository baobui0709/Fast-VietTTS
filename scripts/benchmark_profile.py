#!/usr/bin/env python3
"""Run Fast-VietTTS benchmark/profile suite."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.profiler import run_benchmark_suite, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Fast-VietTTS")
    parser.add_argument("--ref", default=None, help="Optional reference voice wav")
    parser.add_argument("--output", default="benchmark_outputs", help="Output directory")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat per case")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = run_benchmark_suite(
        output_dir=args.output,
        reference_audio=args.ref,
        repeat=args.repeat,
        device=args.device,
    )
    save_results(
        results,
        json_path=os.path.join(args.output, "profile_results.json"),
        md_path="docs/PROFILE.md",
    )
    print("✅ Benchmark complete")
    print("- JSON:", os.path.join(args.output, "profile_results.json"))
    print("- Markdown: docs/PROFILE.md")


if __name__ == "__main__":
    main()
