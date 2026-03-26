# 01 Combined All

**Base**: SOTA #1 (LeakyReLU² + TTT + Parallel Muon, 1.1194 BPB)
**Target**: 1.1100-1.1150 BPB

## Technique

The SOTA #1 script already combines all proven techniques: LeakyReLU(0.5)², Legal Score-First TTT, Parallel Muon with Parameter Banking, GPTQ-lite (5-percentile search), EMA (0.997), XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale, SmearGate, BigramHash, int6+lzma compression.

This experiment adds noisy QAT calibrated to int6 precision (from depth recurrence research) as an optional enhancement.

## Changes from Base

- Added `NOISY_QAT=1` env var: injects calibrated noise `(rand-0.5) * amax/31.0` matching int6 precision during late QAT, instead of STE fake-quantize

## Run

```bash
SEED=1337 bash run.sh
# With noisy QAT: SEED=1337 NOISY_QAT=1 bash run.sh
# With W&B: SEED=1337 WANDB_ENABLED=1 bash run.sh
```

## Results

| Seed | Variant | BPB | BPB (RT) | Artifact | Steps | Time |
|------|---------|-----|----------|----------|-------|------|
| 1337 | STE QAT | -- | -- | -- | -- | -- |
| 1337 | Noisy QAT | -- | -- | -- | -- | -- |
| 42   | STE QAT | -- | -- | -- | -- | -- |
| 2024 | STE QAT | -- | -- | -- | -- | -- |
