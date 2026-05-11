# Fast-VietTTS

Fast-VietTTS là repo triển khai ViterBox-compatible Text-to-Speech tiếng Việt, tập trung vào **chạy ổn định trên Google Colab**, **Gradio public link/API**, **batch audiobook**, và **giữ chất lượng ViterBox gốc**.

## Trạng thái hiện tại

Fast-VietTTS hiện tại đã ổn định cho sử dụng thực tế:

- Dùng ViterBox core vendored trong repo.
- Đọc số / viết tắt tiếng Việt đúng nhờ `soe-vinorm`.
- Giữ pipeline gốc của ViterBox: `split_sentences=True`, VAD, fade/crossfade.
- Có Gradio app với public link.
- Có batch generator cho nhiều chương.
- Có cache reference conditioning cho một giọng cố định / cùng một voice.

## Điều quan trọng

Dự án hiện tại **không phải Turbo model** và **không nhanh hơn ViterBox gốc nhiều lần**. Các benchmark đã cho thấy bottleneck chính nằm ở `T3 inference / LlamaModel.forward` autoregressive token-by-token. Wrapper-level optimization gần như đã hết dư địa nếu vẫn giữ chất lượng tự nhiên.

Điểm tối ưu đáng giữ:

- Reference conditioning cache: lần đầu với voice mới sẽ chậm hơn, các lần sau cùng voice nhanh hơn rõ.

Các hướng đã test nhưng không giữ:

- `split_sentences=False`: nhanh hơn nhẹ nhưng nghe kém tự nhiên hơn.
- Tắt VAD/fade: tăng tốc nhỏ, không đáng rủi ro.
- Tắt alignment analyzer: tăng tốc nhỏ, không đáng rủi ro.
- `output_attentions=False`: không nhanh hơn trong test.
- `torch.compile` mặc định: không cải thiện ổn định.
- Fixed voice preload riêng: không cải thiện tổng thời gian.

## Cài đặt Colab sạch

Khuyến nghị dùng runtime sạch, GPU T4, sau đó chạy:

```python
%cd /content
!rm -rf Fast-VietTTS
!git clone https://github.com/baobui0709/Fast-VietTTS.git
%cd Fast-VietTTS
```

Cài dependencies theo môi trường ViterBox đã test:

```python
!pip install -q --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q --no-cache-dir transformers==4.46.3 diffusers==0.29.0 safetensors==0.5.3     einops omegaconf librosa==0.11.0 scipy soundfile gradio==5.44.1     soe-vinorm conformer==0.3.2 s3tokenizer resemble-perth spacy-pkuseg pykakasi
```

Sau khi cài xong nên restart runtime.

## Test normalizer tiếng Việt

```python
%cd /content/Fast-VietTTS
from viterbox.tts import HAS_VINORM, normalize_text
print('HAS_VINORM:', HAS_VINORM)
print(normalize_text('Tôi có 3 người bạn. Năm 2025 tôi đến TP.HCM.', 'vi'))
```

Kỳ vọng:

```text
HAS_VINORM: True
Tôi có ba người bạn . Năm hai nghìn không trăm hai mươi lăm tôi đến Thành phố Hồ Chí Minh .
```

## Quick Start Python

```python
from src.engine import FastVietTTS
from IPython.display import Audio

tts = FastVietTTS(device='cuda', use_fp16=False, compile_model=False)

audio = tts.generate(
    text='Tôi có 3 người bạn. Năm 2025 tôi đến TP.HCM.',
    language='vi',
    audio_prompt='wavs/00c3bca9-4885-4b2e-8139-10e09df6143c.wav',
)

tts.save_audio(audio, '/content/test.wav')
Audio('/content/test.wav')
```

## Gradio app

```python
%cd /content/Fast-VietTTS
!python app.py
```

Sau khi chạy, Gradio sẽ in public URL dạng:

```text
https://xxxxx.gradio.live
```

Ứng dụng ngoài có thể gọi Gradio public API bằng `gradio_client`. Nên kiểm tra endpoint bằng:

```python
!pip install -q gradio_client
from gradio_client import Client
client = Client('https://xxxxx.gradio.live')
print(client.view_api())
```

## Batch audiobook

```bash
python scripts/batch_from_txt.py   --input /content/story.txt   --ref wavs/00c3bca9-4885-4b2e-8139-10e09df6143c.wav   --output /content/batch_output   --device cuda   --resume
```

## Phase model-level

Nếu muốn nhanh hơn nhiều lần mà không làm giọng robot, cần sang hướng model-level:

1. Audit dataset `metadata.csv` dạng `file.wav|text`.
2. Upload dataset / token shards lên Hugging Face.
3. Train kiểu relay checkpoint trên Colab T4 theo từng phiên.
4. Ưu tiên distill student T3 nhỏ cho một giọng cố định, giữ S3Gen gốc càng lâu càng tốt.

Xem thêm:

- `docs/SPEC.md`
- `docs/AGENTS.md`
- `docs/COLAB_CLEAN_RUN.md`
- `docs/HF_RELAY_TRAINING_PLAN.md`
