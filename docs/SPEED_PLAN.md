# Speed Optimization — Agent 10

## Goal

Reduce current ViterBox fallback RTF from about `1.14x` to below `0.8x` if possible without quality loss.

## Safe optimizations

1. Enable TF32 on CUDA.
2. Warm up model once after load.
3. Benchmark selective `torch.compile` on `t3`, `s3gen`, or both.
4. Keep `use_fp16=False` because fallback FP16 caused dtype mismatch.
5. Keep ViterBox text pipeline unchanged because `soe-vinorm` is required for Vietnamese number normalization.

## Commands

Baseline safe optimization:

```bash
python scripts/speed_benchmark.py --ref wavs/00c3bca9-4885-4b2e-8139-10e09df6143c.wav
```

Try compile variants:

```bash
python scripts/speed_benchmark.py   --ref wavs/00c3bca9-4885-4b2e-8139-10e09df6143c.wav   --try-compile
```

## Outputs

- `benchmark_outputs/speed_results.json`
- `docs/SPEED_RESULTS.md`

## Decision rule

Only keep an optimization if:

- RTF improves consistently.
- Audio quality is unchanged by listening test.
- No CUDA OOM or compile crash on Colab T4.
