# Colab Clean Run — Fast-VietTTS

Tài liệu này dùng sau khi `Disconnect and delete runtime` để chạy lại repo từ môi trường sạch.

## 1. Clone repo

```python
%cd /content
!rm -rf Fast-VietTTS
!git clone https://github.com/baobui0709/Fast-VietTTS.git
%cd Fast-VietTTS
```

## 2. Cài dependencies

```python
!pip install -q --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q --no-cache-dir transformers==4.46.3 diffusers==0.29.0 safetensors==0.5.3     einops omegaconf librosa==0.11.0 scipy soundfile gradio==5.44.1     soe-vinorm conformer==0.3.2 s3tokenizer resemble-perth spacy-pkuseg pykakasi
```

Sau khi cài xong, nên restart runtime.

## 3. Kiểm tra GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

## 4. Kiểm tra `soe-vinorm`

```python
from viterbox.tts import HAS_VINORM, normalize_text
print('HAS_VINORM:', HAS_VINORM)
print(normalize_text('Tôi có 3 người bạn. Năm 2025 tôi đến TP.HCM.', 'vi'))
```

## 5. Test Python inference

```python
from src.engine import FastVietTTS
from IPython.display import Audio

tts = FastVietTTS(device='cuda', use_fp16=False, compile_model=False)
audio = tts.generate(
    text='Tôi có 3 người bạn. Năm 2025 tôi đến TP.HCM.',
    language='vi',
    audio_prompt='wavs/00c3bca9-4885-4b2e-8139-10e09df6143c.wav',
)
tts.save_audio(audio, '/content/clean_test.wav')
Audio('/content/clean_test.wav')
```

## 6. Test Gradio

```python
!python app.py
```

Sau đó lấy public URL và kiểm tra API:

```python
!pip install -q gradio_client
from gradio_client import Client
client = Client('https://xxxxx.gradio.live')
print(client.view_api())
```

## 7. Test batch

```python
%%writefile /content/test_story.txt
Chương 1
Tôi có 3 người bạn. Năm 2025 tôi đến TP.HCM.

Chương 2
Đây là chương thứ hai dùng cùng một giọng mẫu.
```

```python
!python scripts/batch_from_txt.py   --input /content/test_story.txt   --ref wavs/00c3bca9-4885-4b2e-8139-10e09df6143c.wav   --output /content/batch_test   --device cuda
```
