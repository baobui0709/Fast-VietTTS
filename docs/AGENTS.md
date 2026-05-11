# AGENTS — Fast-VietTTS Implementation Log

## Agent 0 — Compatibility

Kết luận: Turbo không tương thích trực tiếp với `AnhTuan89/viterbox`. Dùng fallback ViterBox architecture.

## Agent 1 — Core Engine

Tạo `src/engine.py`, wrapper `FastVietTTS`.

## Agent 2 — Text Processor

Tạo text processor ban đầu. Sau đó kết luận: không dùng TextProcessor trong luồng chính khi ViterBox + `soe-vinorm` hoạt động.

## Agent 3 — Audio Utils

Tạo audio utilities.

## Agent 4 — Gradio App

Tạo `app.py` với lazy loading model.

## Agent 5 — Batch Generator

Tạo `src/batch_generator.py` và CLI batch từ `.txt`.

## Agent 6 — Colab Notebooks

Tạo notebooks QuickStart, VoiceClone, BatchChapter, Benchmark.

## Agent 7 — README / Benchmark Docs

Tạo README và benchmark template.

## Agent 8 — Vendor ViterBox Core

Copy `viterbox/` và `wavs/` vào repo để standalone hơn.

## Fix — `soe-vinorm`

Bổ sung dependency `soe-vinorm`. Đây là fix quan trọng để ViterBox normalize số/viết tắt tiếng Việt đúng.

## Agent 9 — Benchmark & Profile

Đo RTF baseline, xác định tốc độ thường quanh `1.1x–1.3x` tùy text/runtime.

## Agent 10 — Speed Benchmark

Test TF32, warmup, `torch.compile`. Không bật compile mặc định vì không cải thiện ổn định.

## Agent 11 / 11B — Deep Profile

Xác định các bottleneck:

- `T3.inference`
- `S3Gen.inference`
- `prepare_conditionals`

## Agent 12 — Reference Conditioning Cache

Giữ lại. Tối ưu đáng giá nhất ở wrapper-level.

Kết quả test:

```text
Run 1: 14.00s
Run 2: 6.83s
Run 3: 7.54s
Speedup after cache: 1.95x
```

## Agent 13 — Gradio / Batch Cache Validation

Không cần patch. Gradio lazy-load model, Batch reuse engine nên Agent 12 cache hoạt động.

## Agent 14 — VAD/Post-processing Benchmark

Không giữ. Tăng tốc quá nhỏ.

## Agent 15 — Split Strategy Benchmark

Không giữ. `split_off` nhanh hơn nhẹ nhưng nghe kém tự nhiên hơn. Giữ `split_sentences=True`.

## Agent 16 — Fixed Voice Preload

Không giữ. Không cải thiện tổng thời gian.

## Agent 17 — T3 Sampling Inspect

Xác định bottleneck thật:

```text
T3.inference
LlamaModel.forward autoregressive nhiều lần
```

## Agent 18 — Alignment Analyzer Benchmark

Không giữ. Tăng tốc nhỏ, không đáng rủi ro.

## Agent 19 — Output Attentions Benchmark

Không giữ. `output_attentions=False` không nhanh hơn, baseline nghe ổn hơn.

## Phase tiếp theo — Model-level

Hướng phù hợp với Colab T4:

- HF dataset / token shards.
- Relay training checkpoint qua Hugging Face.
- Một giọng cố định.
- Student T3 nhỏ, giữ S3Gen gốc nếu có thể.
