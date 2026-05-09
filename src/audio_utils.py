import os
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch
import torchaudio


AudioResult = Union[str, torch.Tensor]


def prepare_reference(
    input_path: str,
    output_path: Optional[str] = None,
    target_sr: int = 24000,
    target_duration: float = 12.0,
    trim_silence: bool = True,
    normalize: bool = True,
) -> AudioResult:
    """Tiền xử lý reference audio cho voice cloning."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Reference audio not found: {input_path}")

    try:
        audio, sr = librosa.load(input_path, sr=None, mono=True)
    except Exception as exc:
        raise ValueError(f"Unsupported or unreadable audio file: {input_path}") from exc

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {input_path}")

    duration = audio.shape[0] / float(sr)
    already_good = (
        output_path is None
        and sr == target_sr
        and 5.0 <= duration <= 20.0
        and audio.ndim == 1
    )

    if already_good:
        tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        return tensor

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=20)

    if audio.size == 0:
        raise ValueError("Audio became empty after silence trimming")

    max_samples = int(target_duration * target_sr)
    if audio.shape[0] > max_samples:
        audio = audio[:max_samples]

    if normalize:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak * 0.95

    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    if output_path is None:
        return tensor

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), tensor, target_sr)
    return str(out_path)


def crossfade_concat(
    audio_list: list[torch.Tensor],
    crossfade_ms: int = 50,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """Ghép list audio tensors với crossfade mượt mà."""
    if not audio_list:
        return torch.zeros(1, 0)

    if len(audio_list) == 1:
        return _ensure_2d(audio_list[0])

    fade_samples = max(0, int(sample_rate * crossfade_ms / 1000))
    output = _ensure_2d(audio_list[0]).clone()

    for next_audio in audio_list[1:]:
        next_audio = _ensure_2d(next_audio)
        overlap = min(fade_samples, output.shape[-1], next_audio.shape[-1])

        if overlap <= 0:
            output = torch.cat([output, next_audio], dim=-1)
            continue

        fade_out = torch.linspace(1.0, 0.0, overlap, device=output.device).view(1, -1)
        fade_in = torch.linspace(0.0, 1.0, overlap, device=next_audio.device).view(1, -1)

        mixed = output[:, -overlap:] * fade_out + next_audio[:, :overlap] * fade_in
        output = torch.cat([output[:, :-overlap], mixed, next_audio[:, overlap:]], dim=-1)

    return output


def add_silence(duration_ms: int, sample_rate: int = 24000) -> torch.Tensor:
    """Tạo tensor im lặng shape (1, N)."""
    if duration_ms < 0:
        raise ValueError("duration_ms must be >= 0")

    samples = int(sample_rate * duration_ms / 1000)
    return torch.zeros(1, samples)


def get_audio_duration(tensor: torch.Tensor, sample_rate: int = 24000) -> float:
    """Trả về độ dài audio tính bằng giây."""
    tensor = _ensure_2d(tensor)
    return tensor.shape[-1] / float(sample_rate)


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Đảm bảo tensor audio có shape (1, N)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("audio must be a torch.Tensor")

    if tensor.ndim == 1:
        return tensor.unsqueeze(0)

    if tensor.ndim == 2:
        if tensor.shape[0] == 1:
            return tensor
        return tensor.mean(dim=0, keepdim=True)

    raise ValueError("audio tensor must have shape (N,) or (C, N)")


if __name__ == "__main__":
    a = torch.ones(1, 24000)
    b = torch.zeros(1, 24000)
    silence = add_silence(500)
    joined = crossfade_concat([a, silence, b], crossfade_ms=50)

    assert silence.shape == (1, 12000)
    assert joined.ndim == 2 and joined.shape[0] == 1
    assert abs(get_audio_duration(a) - 1.0) < 1e-6

    print("✅ audio_utils tests passed")
