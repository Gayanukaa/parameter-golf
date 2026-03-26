# 07 LR and Warmdown

**Base**: Best from 01_combined_all
**Target**: 0.001-0.003 BPB each variant

## Technique

Three near-trivial changes to the learning rate schedule and QAT noise. All current submissions use linear warmdown. Cosine may settle into flatter minima. Longer warmdown gives more time for EMA/SWA to collect good checkpoints. Noisy QAT calibrated to int6 precision reduces quantization gap.

## Changes from Base

- `LR_SCHEDULE=cosine`: cosine decay during warmdown instead of linear
- `WARMDOWN_ITERS=4500`: longer warmdown (vs 3500)
- `NOISY_QAT=1`: noise = (rand-0.5) * amax/31.0 calibrated to int6

## Run

```bash
# Test cosine schedule
LR_SCHEDULE=cosine SEED=1337 bash run.sh

# Test linear (baseline behavior) with longer warmdown
LR_SCHEDULE=linear SEED=1337 bash run.sh
```

## Results

| Variant | Seed | BPB | BPB (RT) | Artifact | Steps | Time |
|---------|------|-----|----------|----------|-------|------|
| cosine  | 1337 | --  | --       | --       | --    | --   |
| linear  | 1337 | --  | --       | --       | --    | --   |
