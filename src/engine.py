import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio

from src.audio_utils import add_silence, crossfade_concat
from src.text_processor import TextProcessor


SAMPLE_RATE = 24000


class FastVietTTS:
    """ViterBox-compatible TTS wrapper using fallback ViterBox architecture."""

    def __init__(
        self,
        device: str = "cuda",
        use_fp16: bool = False,
        compile_model: bool = True,
    ):
        """Load model and apply safe inference optimizations."""
        self.device = self._resolve_device(device)
        self.use_fp16 = bool(use_fp16 and self.device == "cuda")
        self.compile_model = bool(compile_model)
        self.text_processor = TextProcessor()
        self._ref_cache = {}
        self.model = self._load_model()
        self._force_eager_attention()

        if self.use_fp16:
            self._apply_fp16()

        if self.compile_model:
            self._compile_model()

    def generate(
        self,
        text: str,
        language: str = "vi",
        audio_prompt: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        sentence_pause_ms: int = 500,
        crossfade_ms: int = 50,
    ) -> torch.Tensor:
        """Generate speech using original ViterBox pipeline with reference cache."""
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        if audio_prompt and not os.path.exists(audio_prompt):
            raise FileNotFoundError(f"audio_prompt not found: {audio_prompt}")

        try:
            with torch.inference_mode():
                prepared_prompt = self._ensure_reference_conditioning(
                    audio_prompt,
                    exaggeration,
                )
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
                )
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError("CUDA out of memory. Try shorter text or CPU mode.") from exc

        return self._ensure_audio(wav).cpu()

    def save_audio(self, audio: torch.Tensor, path: str) -> None:
        """Save audio tensor to WAV file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), self._ensure_audio(audio).cpu(), SAMPLE_RATE)

    @classmethod
    def from_pretrained(cls, device: str = "cuda") -> "FastVietTTS":
        """Create FastVietTTS instance."""
        return cls(device=device)

    def _force_eager_attention(self) -> None:
        """Force eager attention for ViterBox alignment analyzer."""
        try:
            self.model.t3.cfg._attn_implementation = "eager"
            self.model.t3.tfmr.config._attn_implementation = "eager"
        except Exception:
            pass

    def _load_model(self) -> Any:
        """Load fallback ViterBox model."""
        try:
            from viterbox import Viterbox
        except Exception as exc:
            raise ImportError(
                "ViterBox package is required for fallback mode. "
                "Install original TienHiep-TTS/viterbox before running inference."
            ) from exc

        try:
            return Viterbox.from_pretrained(self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load ViterBox on {self.device}") from exc

    def _ensure_reference_conditioning(
        self,
        audio_prompt: Optional[str],
        exaggeration: float,
    ) -> Optional[str]:
        """Prepare ViterBox reference conditioning once per reference hash."""
        if not audio_prompt:
            return None

        key = f"{self._file_md5(audio_prompt)}:{float(exaggeration):.4f}"

        if self._ref_cache.get("active_key") != key:
            self.model.prepare_conditionals(audio_prompt, exaggeration)
            self._ref_cache["active_key"] = key
            self._ref_cache[key] = True

        return None

    def _generate_one(self, text: str, **kwargs: Any) -> torch.Tensor:
        """Call original ViterBox.generate API."""
        return self.model.generate(
            text=text,
            language=kwargs.get("language", "vi"),
            audio_prompt=kwargs.get("audio_prompt"),
            exaggeration=kwargs.get("exaggeration", 0.5),
            cfg_weight=kwargs.get("cfg_weight", 0.5),
            temperature=kwargs.get("temperature", 0.8),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 2.0),
            sentence_pause_ms=kwargs.get("sentence_pause_ms", 500),
            crossfade_ms=kwargs.get("crossfade_ms", 50),
        )

    def _apply_fp16(self) -> None:
        """Apply FP16 to available modules."""
        for module in self._candidate_modules():
            try:
                module.half()
            except Exception:
                pass

    def _compile_model(self) -> None:
        """Compile available submodules with safe fallback."""
        if not hasattr(torch, "compile"):
            return

        for parent, name, module in self._named_candidate_modules():
            try:
                setattr(parent, name, torch.compile(module, mode="reduce-overhead"))
            except Exception:
                pass

    def _candidate_modules(self) -> list[Any]:
        """Return modules that may support half()."""
        modules = []

        for obj in [self.model, getattr(self.model, "model", None)]:
            if obj is not None and hasattr(obj, "half"):
                modules.append(obj)

        for name in ("t3", "s3gen", "ve", "voice_encoder", "decoder"):
            obj = getattr(self.model, name, None)
            if obj is not None and hasattr(obj, "half"):
                modules.append(obj)

        return modules

    def _named_candidate_modules(self) -> list[tuple[Any, str, Any]]:
        """Return submodules that may support torch.compile."""
        found = []

        for parent in [self.model, getattr(self.model, "model", None)]:
            if parent is None:
                continue

            for name in ("t3", "s3gen", "ve", "voice_encoder", "decoder"):
                module = getattr(parent, name, None)
                if module is not None and hasattr(module, "forward"):
                    found.append((parent, name, module))

        return found

    def _cache_reference(self, audio_prompt: str) -> str:
        """Cache reference audio path by md5."""
        key = self._file_md5(audio_prompt)
        if key not in self._ref_cache:
            self._ref_cache[key] = audio_prompt
        return self._ref_cache[key]

    @staticmethod
    def _file_md5(path: str) -> str:
        """Return file md5 hash."""
        md5 = hashlib.md5()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                md5.update(block)
        return md5.hexdigest()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device safely."""
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    @staticmethod
    def _ensure_audio(audio: torch.Tensor) -> torch.Tensor:
        """Ensure audio shape is (1, N)."""
        if not isinstance(audio, torch.Tensor):
            raise TypeError("audio must be a torch.Tensor")

        if audio.ndim == 1:
            return audio.unsqueeze(0)

        if audio.ndim == 2:
            if audio.shape[0] == 1:
                return audio
            return audio.mean(dim=0, keepdim=True)

        raise ValueError("audio tensor must have shape (N,) or (C, N)")


if __name__ == "__main__":
    tts = FastVietTTS.from_pretrained(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    audio = tts.generate("Xin chào! Đây là kiểm tra Fast-VietTTS.")
    tts.save_audio(audio, "/tmp/test_output.wav")
    print(f"✅ Audio saved: shape={tuple(audio.shape)}, sr={SAMPLE_RATE}")
