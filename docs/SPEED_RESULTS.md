# Speed Benchmark — Agent 10

| Variant | Case | Chars | Audio sec | Elapsed sec | RTF | Compiled |
|---|---|---:|---:|---:|---:|---|
| tf32_warmup | short | 52 | 3.60 | 5.32 | 1.48 |  |
| tf32_warmup | medium | 117 | 6.40 | 7.07 | 1.10 |  |
| tf32_warmup | long | 212 | 14.26 | 15.26 | 1.07 |  |
| compile_t3 | short | 52 | 3.96 | 4.59 | 1.16 | t3 |
| compile_t3 | medium | 117 | 6.83 | 7.86 | 1.15 | t3 |
| compile_t3 | long | 212 | 13.42 | 14.61 | 1.09 | t3 |
| compile_s3gen | short | 52 | 3.80 | 4.44 | 1.17 | s3gen |
| compile_s3gen | medium | 117 | 6.60 | 7.81 | 1.18 | s3gen |
| compile_s3gen | long | 212 | 13.32 | 14.13 | 1.06 | s3gen |
| compile_t3_s3gen | short | 52 | 3.86 | 5.29 | 1.37 | t3,s3gen |
| compile_t3_s3gen | medium | 117 | 6.64 | 7.25 | 1.09 | t3,s3gen |
| compile_t3_s3gen | long | 212 | 13.87 | 15.65 | 1.13 | t3,s3gen |
