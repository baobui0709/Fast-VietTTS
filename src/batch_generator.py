import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.engine import FastVietTTS
except ImportError:
    from engine import FastVietTTS


class BatchGenerator:
    """Sinh audio hàng loạt từ file text theo từng chương."""

    def __init__(self, engine: FastVietTTS, output_dir: str):
        """Khởi tạo batch generator với engine và thư mục output."""
        self.engine = engine
        self.output_dir = output_dir
        self.index_file = os.path.join(output_dir, "progress.json")
        os.makedirs(output_dir, exist_ok=True)

    def generate_from_file(
        self,
        txt_path: str,
        reference_audio: str | None,
        chapter_split_pattern: str = r"Chương \d+",
        exaggeration: float = 0.4,
        cfg_weight: float = 0.6,
        resume: bool = True,
    ) -> list[str]:
        """Đọc file .txt và sinh audio từng chương với resume support."""
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Text file not found: {txt_path}")

        if reference_audio and not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        text = Path(txt_path).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Text file is empty: {txt_path}")

        chapters = self._split_chapters(text, chapter_split_pattern)
        progress = self._load_progress() if resume else {"done": {}}
        done = progress.setdefault("done", {})

        outputs = []
        total_start = time.time()
        generated_count = 0
        total_audio_sec = 0.0

        for idx, chapter in enumerate(tqdm(chapters, desc="Generating chapters"), start=1):
            chapter_id = f"chapter_{idx:04d}"
            out_path = os.path.join(self.output_dir, f"{chapter_id}.wav")

            if resume and chapter_id in done and os.path.exists(done[chapter_id].get("path", "")):
                outputs.append(done[chapter_id]["path"])
                continue

            start = time.time()

            audio = self.engine.generate(
                text=chapter,
                language="vi",
                audio_prompt=reference_audio,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            self.engine.save_audio(audio, out_path)

            elapsed = time.time() - start
            audio_sec = audio.shape[-1] / 24000
            total_audio_sec += audio_sec
            generated_count += 1

            done[chapter_id] = {
                "path": out_path,
                "chars": len(chapter),
                "audio_sec": audio_sec,
                "elapsed_sec": elapsed,
                "rtf": elapsed / max(audio_sec, 1e-6),
            }

            self._save_progress(progress)
            outputs.append(out_path)

        total_elapsed = time.time() - total_start

        print("✅ Batch complete")
        print(f"Chapters total: {len(chapters)}")
        print(f"Generated this run: {generated_count}")
        print(f"Elapsed: {total_elapsed:.2f}s")

        if total_audio_sec > 0:
            print(f"Audio generated: {total_audio_sec:.2f}s")
            print(f"Average RTF: {total_elapsed / total_audio_sec:.2f}x")

        return outputs

    def _split_chapters(self, text: str, pattern: str) -> list[str]:
        """Tách text thành danh sách chương bằng regex heading."""
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

        if not matches:
            parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            return parts or [text]

        chapters = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chapter = text[start:end].strip()
            if chapter:
                chapters.append(chapter)

        prefix = text[:matches[0].start()].strip()
        if prefix:
            chapters.insert(0, prefix)

        return chapters

    def _load_progress(self) -> dict[str, Any]:
        """Load progress.json nếu tồn tại."""
        if not os.path.exists(self.index_file):
            return {"done": {}}

        try:
            return json.loads(Path(self.index_file).read_text(encoding="utf-8"))
        except Exception:
            return {"done": {}}

    def _save_progress(self, progress: dict[str, Any]) -> None:
        """Lưu progress.json."""
        Path(self.index_file).write_text(
            json.dumps(progress, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    class DummyEngine:
        """Engine giả để test nhanh không cần model."""

        def generate(self, text: str, **kwargs: Any):
            import torch
            return torch.zeros(1, 2400)

        def save_audio(self, audio, path: str) -> None:
            Path(path).write_bytes(b"dummy")

    tmp = Path("/tmp/fast_viettts_batch_test")
    tmp.mkdir(exist_ok=True)

    txt = tmp / "test.txt"
    txt.write_text("Chương 1\nXin chào.\n\nChương 2\nTạm biệt.", encoding="utf-8")

    gen = BatchGenerator(DummyEngine(), str(tmp / "out"))
    files = gen.generate_from_file(str(txt), reference_audio=None, resume=False)

    assert len(files) == 2
    assert all(Path(f).exists() for f in files)

    print("✅ batch_generator tests passed")
