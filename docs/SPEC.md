# SPEC — Fast-VietTTS

## Mục tiêu

Fast-VietTTS hướng đến:

1. Giữ chất lượng / độ tự nhiên của ViterBox gốc.
2. Chạy ổn định trên Google Colab T4.
3. Có Gradio public link/API để ứng dụng ngoài gọi.
4. Hỗ trợ batch tạo audiobook nhiều chương.
5. Tối ưu tốc độ nhưng không đánh đổi chất lượng.
6. Chuẩn bị phase model-level cho training dần trên Colab T4.

## Phạm vi hiện tại

Fast-VietTTS hiện là:

```text
ViterBox core vendored
+ FastVietTTS wrapper
+ Gradio app
+ Batch generator
+ Colab notebooks
+ reference conditioning cache
```

## Yêu cầu chất lượng

Không được đánh đổi chất lượng bằng các cách sau làm mặc định:

- Không tắt `split_sentences` mặc định.
- Không tắt VAD/fade/crossfade mặc định.
- Không dùng fast mode nghe robot.
- Không tự normalize số thay ViterBox khi `soe-vinorm` đã hoạt động.
- Không bỏ pipeline ViterBox gốc nếu chưa chứng minh chất lượng tương đương.

## Yêu cầu tiếng Việt

- `soe-vinorm` phải được cài.
- `HAS_VINORM` phải là `True`.
- Số/năm/viết tắt như `3`, `2025`, `TP.HCM` phải được normalize đúng bởi ViterBox pipeline.

## Yêu cầu runtime

- GPU khuyến nghị: Colab T4.
- `use_fp16=False` mặc định vì fallback từng gặp dtype mismatch.
- `compile_model=False` mặc định vì benchmark không cho lợi ích ổn định.

## Yêu cầu Gradio API

- `app.py` phải lazy-load model.
- Public link Gradio dùng được khi Colab runtime còn sống.
- API input/output cần kiểm tra bằng `gradio_client.Client(...).view_api()` trước khi tích hợp app ngoài.

## Yêu cầu batch

- Batch không tạo engine mới trong vòng lặp chương.
- Cùng một reference voice phải hưởng cache conditioning từ Agent 12.
- Có resume bằng `progress.json`.

## Kết luận tốc độ hiện tại

Fast-VietTTS không nhanh hơn ViterBox nhiều lần. Lợi ích chính là:

- ổn định hơn trong workflow Colab;
- cache reference conditioning cho cùng một voice;
- batch/app reuse model.

Bottleneck chính còn lại:

```text
T3 inference / LlamaModel.forward autoregressive token-by-token
```
