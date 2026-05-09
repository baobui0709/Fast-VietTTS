# Fast-VietTTS

Fast-VietTTS là dự án nâng cấp ViterBox/TienHiep-TTS cho Text-to-Speech tiếng Việt và voice cloning, chạy trên Google Colab và GitHub.

> Kết quả compatibility hiện tại: `AnhTuan89/viterbox` **không tương thích trực tiếp** với Chatterbox-Turbo. Dự án đang dùng fallback: ViterBox architecture gốc + tối ưu inference an toàn.

## Tính năng

- TTS tiếng Việt và English thông qua ViterBox-compatible API.
- Zero-shot voice cloning với audio mẫu.
- Text normalization tiếng Việt: viết tắt, số, dấu câu, chunking.
- Audio utilities: resample, mono, trim silence, normalize, crossfade.
- Gradio Web UI có device selector và RTF realtime.
- Batch generator từ file `.txt` theo chương, có resume bằng `progress.json`.

## Model Strategy

Do `docs/COMPAT_REPORT.md` kết luận Turbo không compatible, Fast-VietTTS hiện dùng:

1. `AnhTuan89/viterbox` / ViterBox architecture gốc.
2. `torch.inference_mode()` cho generate.
3. `torch.compile` optional, có fallback.
4. `use_fp16=False` mặc định vì FP16 gây dtype mismatch với ViterBox fallback.
5. Text chunking và reference audio cache.

## So sánh với ViterBox

| Hạng mục | ViterBox gốc | Fast-VietTTS hiện tại |
|---|---|---|
| Model weights | `AnhTuan89/viterbox` | `AnhTuan89/viterbox` |
| Turbo decoder | Không | Không, do không compatible |
| Voice clone | Có | Có |
| Text processor tiếng Việt | Cơ bản | Cải thiện |
| Crossfade concat | Có | Có |
| Batch resume | Không rõ | Có |
| RTF UI | Không | Có |
| FP16 | Có thể dùng | Tắt mặc định để tránh lỗi dtype |

## Cài đặt nhanh trên Colab

```python
!git clone https://github.com/baobui0709/Fast-VietTTS.git
%cd Fast-VietTTS
```

```python
!pip install -q --upgrade pip
!pip install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchaudio==2.6.0
!pip install -q numpy==1.26.4 librosa soundfile num2words tqdm huggingface_hub gradio
!pip install -q s3tokenizer omegaconf conformer safetensors
```

Clone ViterBox source fallback:

```python
%cd /content
!git clone https://github.com/yingchunbin/TienHiep-TTS.git
%cd /content/Fast-VietTTS
```

Nếu gặp lỗi `torchvision::nms` hoặc `LlamaModel`, gỡ torchvision:

```python
!pip uninstall -y torchvision torchtext
```

Sau đó restart runtime.

## Quick Start

```python
import sys

viterbox_src = "/content/TienHiep-TTS"
if viterbox_src not in sys.path:
    sys.path.insert(0, viterbox_src)

from src.engine import FastVietTTS

tts = FastVietTTS(device="cuda", use_fp16=False, compile_model=False)
audio = tts.generate("Xin chào! Đây là kiểm tra Fast-VietTTS.")
tts.save_audio(audio, "output.wav")
```

## Voice Cloning

```python
audio = tts.generate(
    text="Tôi có thể nói bằng giọng mẫu của bạn.",
    language="vi",
    audio_prompt="ref.wav",
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    sentence_pause_ms=500,
    crossfade_ms=50,
)
tts.save_audio(audio, "voice_clone.wav")
```

## Gradio App

```python
!python app.py
```

Mở URL Gradio được in ra trong Colab.

## Batch Generation

```bash
python scripts/batch_from_txt.py   --input truyen.txt   --ref ref_voice.wav   --output ./audio_output   --device cuda   --resume
```

Không dùng voice clone:

```bash
python scripts/batch_from_txt.py   --input truyen.txt   --output ./audio_output   --device cuda
```

## API Reference

### `FastVietTTS.from_pretrained(device)`

```python
tts = FastVietTTS.from_pretrained("cuda")
```

### `generate(...)`

```python
audio = tts.generate(
    text="...",
    language="vi",
    audio_prompt=None,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    sentence_pause_ms=500,
    crossfade_ms=50,
)
```

### `save_audio(audio, path)`

```python
tts.save_audio(audio, "output.wav")
```

## Thông số gợi ý cho tiếng Việt

- `exaggeration`: `0.4–0.6`
- `cfg_weight`: `0.5–0.7`
- `temperature`: `0.7–0.9`
- `sentence_pause_ms`: `300–700`
- `crossfade_ms`: `30–80`
- Chunk text: khoảng `150` ký tự/chunk.

## Colab Notebooks

> Đang chuẩn bị trong Agent 6.

- `notebooks/01_QuickStart.ipynb`
- `notebooks/02_VoiceClone.ipynb`
- `notebooks/03_BatchChapter.ipynb`
- `notebooks/04_Benchmark.ipynb`

## Known Issues

- Chatterbox-Turbo không load trực tiếp được `AnhTuan89/viterbox` theo API hiện tại.
- FP16 fallback có thể gây lỗi dtype `Float and Half`, nên mặc định tắt.
- Trên Colab, `torchvision` có thể gây lỗi `LlamaModel`; nên gỡ `torchvision` nếu không dùng.
- RTF hiện tại trên T4 còn khoảng `1.3x` trong test ngắn, chưa đạt mục tiêu realtime.

## License & Credits

- Model: `AnhTuan89/viterbox`, license CC BY-NC 4.0.
- Source tham chiếu: `yingchunbin/TienHiep-TTS` và ViterBox package.
- Framework tham chiếu: Resemble AI Chatterbox.

Dự án này kế thừa và ghi nhận công sức từ cộng đồng ViterBox, TienHiep-TTS, Chatterbox và các dataset tiếng Việt liên quan.
