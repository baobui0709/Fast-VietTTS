# Benchmark — Fast-VietTTS

> Template benchmark. Kết quả đầy đủ sẽ được cập nhật sau Agent 6 / Notebook 04.

## Environment

| Item | Value |
|---|---|
| Runtime | Google Colab |
| GPU | Tesla T4 |
| Torch | 2.6.0+cu124 |
| Torchaudio | 2.6.0+cu124 |
| Model strategy | ViterBox fallback |
| FP16 | Disabled |
| Compile | Disabled for stability |

## Compatibility Result

`AnhTuan89/viterbox` không tương thích trực tiếp với Chatterbox-Turbo. Xem chi tiết tại:

```text
docs/COMPAT_REPORT.md
```

## Initial Functional Test

| Test | Result |
|---|---:|
| Text | `Xin chào! Đây là kiểm tra Fast-VietTTS.` |
| Output shape | `(1, 62640)` |
| Audio duration | `2.61s` |
| Inference elapsed | `3.48s` |
| RTF | `1.33x` |
| Status | Passed |

## Benchmark Plan

Notebook `notebooks/04_Benchmark.ipynb` sẽ benchmark:

1. Text ngắn khoảng 50 ký tự.
2. Text vừa khoảng 200 ký tự.
3. Text dài khoảng 500 ký tự.
4. Có / không reference audio.
5. So sánh ViterBox fallback với các cấu hình tối ưu:
   - baseline FP32
   - optional `torch.compile`
   - chunk size khác nhau

## Target

| Hardware | Target RTF |
|---|---:|
| Colab T4 | `< 0.5` |
| A100 | `< 0.2` |

## Notes

- Kết quả hiện tại chưa phải benchmark cuối.
- FP16 bị tắt do lỗi dtype mismatch trong fallback mode.
- Cần tối ưu tiếp nếu muốn đạt realtime.
