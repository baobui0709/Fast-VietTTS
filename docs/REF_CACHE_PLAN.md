# Reference Conditioning Cache — Agent 12

## Goal
Avoid re-encoding the same reference audio on every generation.

## Why
Deep profile showed `prepare_conditionals` used about 3.9 seconds, around 32% of total time, for a reference-audio generation.

## Apply patch

```bash
python scripts/patch_ref_cache.py
python -m py_compile src/engine.py
```

## Benchmark

```bash
python scripts/benchmark_ref_cache.py --ref wavs/00c3bca9-4885-4b2e-8139-10e09df6143c.wav --repeat 3
```

## Commit rule
Commit only if repeated runs clearly improve after first run and audio quality remains unchanged.
