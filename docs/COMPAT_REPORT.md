# Compatibility Report — Fast-VietTTS

## Final Decision

**NOT COMPATIBLE**

## Summary

`AnhTuan89/viterbox` không thể load trực tiếp bằng Chatterbox-Turbo hiện tại.

## Test Environment

- Device: CUDA
- GPU: Tesla T4
- Torch: 2.6.0+cu124
- Torchaudio: 2.6.0+cu124
- Chatterbox imports: OK

## Key Findings

### 1. Public API

Các class Chatterbox hiện có signature:

- ChatterboxTTS.from_pretrained(device)
- ChatterboxMultilingualTTS.from_pretrained(device)
- ChatterboxTurboTTS.from_pretrained(device)

Public API không nhận model_id nên không thể gọi trực tiếp:

from_pretrained('AnhTuan89/viterbox', device='cuda')

### 2. from_local test

Test với snapshot `AnhTuan89/viterbox`:

- ChatterboxMultilingualTTS.from_local(viterbox_path, 'cuda'): FAILED
- ChatterboxTurboTTS.from_local(viterbox_path, 'cuda'): FAILED

Lỗi chính:

- Multilingual thiếu file: t3_mtl23ls_v2.safetensors
- Turbo thiếu file: ve.safetensors

## Conclusion

`AnhTuan89/viterbox` không tương thích trực tiếp với Chatterbox-Turbo architecture/API hiện tại.

## Model Strategy

Tiếp tục theo fallback:

1. Dùng `AnhTuan89/viterbox` / ViterBox architecture gốc.
2. Không dùng Turbo weights.
3. Tối ưu tốc độ bằng FP16.
4. Dùng torch.inference_mode().
5. Thử torch.compile(mode='reduce-overhead') với fallback.
6. Thêm reference audio cache.
7. Tối ưu text chunking tiếng Việt.

## Status

- T-006: Done
- T-007: Done
