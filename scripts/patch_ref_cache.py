#!/usr/bin/env python3
"""Patch FastVietTTS engine to cache ViterBox reference conditioning."""

from __future__ import annotations

from pathlib import Path

ENGINE = Path("src/engine.py")
BACKUP = Path("src/engine.py.bak_agent12")


def main() -> None:
    if not ENGINE.exists():
        raise FileNotFoundError("src/engine.py not found. Run this from repo root.")

    text = ENGINE.read_text(encoding="utf-8")
    if "_ensure_reference_conditioning" in text:
        print("Agent12 patch already present; nothing to do.")
        return

    BACKUP.write_text(text, encoding="utf-8")

    if "self._ref_cache = {}" in text:
        text = text.replace(
            "self._ref_cache = {}",
            "self._ref_cache = {}  # md5+exaggeration -> prepared ViterBox conds",
            1,
        )

    old_call = """                wav = self._generate_one(
                    text=text.strip(),
                    language=language,
                    audio_prompt=audio_prompt,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    sentence_pause_ms=sentence_pause_ms,
                    crossfade_ms=crossfade_ms,
                )"""

    new_call = """                prepared_prompt = self._ensure_reference_conditioning(audio_prompt, exaggeration)
                wav = self._generate_one(
                    text=text.strip(),
                    language=language,
                    audio_prompt=prepared_prompt,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    sentence_pause_ms=sentence_pause_ms,
                    crossfade_ms=crossfade_ms,
                )"""

    if old_call not in text:
        raise RuntimeError("Expected generate() call block not found. Inspect src/engine.py before patching.")
    text = text.replace(old_call, new_call, 1)

    marker = "    def _generate_one(self, text: str, **kwargs: Any) -> torch.Tensor:"
    if marker not in text:
        raise RuntimeError("_generate_one marker not found.")

    method_lines = ['    def _ensure_reference_conditioning(self, audio_prompt: Optional[str], exaggeration: float) -> Optional[str]:', '        # Prepare ViterBox reference conditioning once per audio hash/exaggeration.', '        if not audio_prompt:', '            return None', '', '        key = f"{self._file_md5(audio_prompt)}:{float(exaggeration):.4f}"', '        if self._ref_cache.get("active_key") != key:', '            self.model.prepare_conditionals(audio_prompt, exaggeration)', '            self._ref_cache["active_key"] = key', '            self._ref_cache[key] = True', '', '        return None', '']
    method = "\n".join(method_lines)
    text = text.replace(marker, method + marker, 1)

    ENGINE.write_text(text, encoding="utf-8")
    print("Patched src/engine.py")
    print("Backup written to src/engine.py.bak_agent12")


if __name__ == "__main__":
    main()
