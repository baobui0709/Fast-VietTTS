#!/usr/bin/env python3
"""CLI sinh audio từng chương từ file txt."""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.batch_generator import BatchGenerator
from src.engine import FastVietTTS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Fast-VietTTS batch generator")

    parser.add_argument("--input", required=True, help="Đường dẫn file .txt")
    parser.add_argument("--ref", default=None, help="Đường dẫn audio mẫu")
    parser.add_argument("--output", required=True, help="Thư mục output")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Thiết bị chạy")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ progress.json")
    parser.add_argument("--pattern", default=r"Chương \d+", help="Regex tách chương")
    parser.add_argument("--exaggeration", type=float, default=0.4)
    parser.add_argument("--cfg-weight", type=float, default=0.6)

    return parser.parse_args()


def main() -> None:
    """Run batch generation."""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    engine = FastVietTTS.from_pretrained(args.device)
    generator = BatchGenerator(engine, args.output)

    files = generator.generate_from_file(
        txt_path=args.input,
        reference_audio=args.ref,
        chapter_split_pattern=args.pattern,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        resume=args.resume,
    )

    print("Generated files:")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
