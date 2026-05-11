# Benchmark — Fast-VietTTS

## Summary

Các benchmark wrapper-level cho thấy bottleneck chính nằm ở:

```text
T3 inference / LlamaModel.forward autoregressive token-by-token
```

## Agent 12 reference cache

Kết quả đáng giữ:

| Run | Elapsed |
|---:|---:|
| 1 | 14.00s |
| 2 | 6.83s |
| 3 | 7.54s |

Speedup sau cache: khoảng `1.95x` cho cùng voice trong test ngắn.

## Agent 15 split strategy

`split_off` nhanh hơn nhẹ nhưng nghe kém tự nhiên hơn. Không giữ.

## Agent 18 alignment

Tắt alignment analyzer tăng khoảng 4% ở medium/long. Không giữ.

## Agent 19 output attentions

`output_attentions=False` không nhanh hơn baseline trong test. Không giữ.

## Practical conclusion

- Giữ pipeline ViterBox gốc.
- Giữ reference conditioning cache.
- Muốn nhanh hơn nhiều cần model-level distillation/student.
