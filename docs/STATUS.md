# STATUS — Fast-VietTTS

## Current stable state

Repo hiện tại đã chạy ổn định cho:

- Python inference.
- Gradio public link/API.
- Batch audiobook.
- Voice reference cố định/cùng voice.
- Đọc số tiếng Việt đúng.

## Stable optimizations kept

- Reference conditioning cache.
- Model reuse trong app/batch.
- ViterBox pipeline gốc.

## Optimizations tested but not kept

- `split_sentences=False`.
- No VAD/fade.
- No alignment analyzer.
- `output_attentions=False`.
- `torch.compile` default.
- Fixed voice preload riêng.

## Practical speed

Bản hiện tại không nhanh hơn ViterBox gốc nhiều lần. Lợi ích chính là reuse reference conditioning khi cùng voice.

## Recommended next action

1. Chạy clean Colab test.
2. Xác nhận Gradio public API với app ngoài.
3. Nếu muốn train lâu dài: sang HF relay training pipeline.
