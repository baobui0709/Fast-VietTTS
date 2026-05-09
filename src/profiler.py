"""Benchmark and profiling helpers for Fast-VietTTS/ViterBox pipeline."""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from src.engine import FastVietTTS


@dataclass
class BenchResult:
    """Single benchmark result."""

    case: str
    chars: int
    use_ref: bool
    audio_sec: float
    elapsed_sec: float
    rtf: float
    output_path: str


TEST_TEXTS = {
    "short": "Xin chào, đây là bài kiểm tra ngắn cho Fast-VietTTS.",
    "medium": (
        "Tiêu Viêm đứng trước cổng học viện, ánh mắt nhìn về phía xa, "
        "lòng tràn đầy quyết tâm bước vào con đường tu luyện mới."
    ),
    "long": (
        "Việt Nam là một quốc gia nằm ở Đông Nam Á với lịch sử lâu đời, văn hóa phong phú "
        "và con người thân thiện. Công nghệ trí tuệ nhân tạo đang mở ra nhiều cơ hội mới "
        "cho việc tạo sách nói, trợ lý giọng nói và hỗ trợ người dùng tiếp cận tri thức "
        "qua giọng nói tự nhiên."
    ),
    "numbers": "Tôi có 3 người bạn. Năm 2025 tôi đến TP.HCM.",
}


def cuda_cleanup() -> None:
    """Release Python and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_generate(
    engine: FastVietTTS,
    text: str,
    case: str,
    output_dir: str,
    reference_audio: Optional[str] = None,
    repeat: int = 1,
) -> list[BenchResult]:
    """Benchmark one text case, optionally with reference audio."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: list[BenchResult] = []

    for run_idx in range(1, repeat + 1):
        cuda_cleanup()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        audio = engine.generate(
            text=text,
            language="vi",
            audio_prompt=reference_audio,
            sentence_pause_ms=500,
            crossfade_ms=50,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        audio_sec = audio.shape[-1] / 24000
        suffix = "ref" if reference_audio else "noref"
        out_path = os.path.join(output_dir, f"{case}_{suffix}_run{run_idx}.wav")
        engine.save_audio(audio, out_path)

        results.append(
            BenchResult(
                case=f"{case}_{suffix}_run{run_idx}",
                chars=len(text),
                use_ref=reference_audio is not None,
                audio_sec=audio_sec,
                elapsed_sec=elapsed,
                rtf=elapsed / max(audio_sec, 1e-6),
                output_path=out_path,
            )
        )

    return results


def run_benchmark_suite(
    output_dir: str = "benchmark_outputs",
    reference_audio: Optional[str] = None,
    repeat: int = 1,
    device: str = "cuda",
) -> list[BenchResult]:
    """Run benchmark suite across predefined texts."""
    engine = FastVietTTS(device=device, use_fp16=False, compile_model=False)
    all_results: list[BenchResult] = []

    # Warmup
    _ = engine.generate("Xin chào.", language="vi", audio_prompt=reference_audio)
    cuda_cleanup()

    for case, text in TEST_TEXTS.items():
        all_results.extend(
            benchmark_generate(
                engine=engine,
                text=text,
                case=case,
                output_dir=output_dir,
                reference_audio=None,
                repeat=repeat,
            )
        )
        if reference_audio:
            all_results.extend(
                benchmark_generate(
                    engine=engine,
                    text=text,
                    case=case,
                    output_dir=output_dir,
                    reference_audio=reference_audio,
                    repeat=repeat,
                )
            )

    return all_results


def save_results(results: list[BenchResult], json_path: str, md_path: str) -> None:
    """Save benchmark results to JSON and Markdown."""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)

    data = [asdict(r) for r in results]
    Path(json_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Profile Results — Fast-VietTTS",
        "",
        "| Case | Chars | Ref | Audio sec | Elapsed sec | RTF |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.case} | {r.chars} | {'yes' if r.use_ref else 'no'} | "
            f"{r.audio_sec:.2f} | {r.elapsed_sec:.2f} | {r.rtf:.2f} |"
        )

    if results:
        avg = sum(r.rtf for r in results) / len(results)
        lines += ["", f"Average RTF: **{avg:.2f}x**"]

    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
