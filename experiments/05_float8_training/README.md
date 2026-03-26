# 05 Float8 Training

**Base**: Best from 01/02/03
**Target**: +48% throughput -> more steps in 600s -> 0.005-0.015 BPB

## Technique

Replace bf16 matmuls with Float8 (e4m3) GEMMs on H100. H100 has native FP8 tensor cores providing ~2x FLOPS. Auto-filter small layers where FP8 overhead exceeds benefit (K*N < 4096^2). Keep backward pass in bf16 for gradient stability.

## Changes from Base

- Replace CastedLinear forward with torch._scaled_mm() for e4m3
- Dynamic per-tensor scaling: scale = amax / fp8_max
- Auto-filter small layers via FLOAT8_MIN_DIM
- Backward stays bf16

## Run

```bash
SEED=1337 bash run.sh
```

## Results

| Seed | BPB | BPB (RT) | Artifact | Steps | Time |
|------|-----|----------|----------|-------|------|
| 1337 | -- | -- | -- | -- | -- |
| 42   | -- | -- | -- | -- | -- |
| 2024 | -- | -- | -- | -- | -- |
