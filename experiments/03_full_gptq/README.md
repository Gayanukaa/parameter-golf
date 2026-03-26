# 03 Full GPTQ

**Base**: Best from 01 or 02
**Target**: 0.005-0.010 BPB (closes 30-50% of quantization gap)

## Technique

Replace GPTQ-lite (5-percentile clip search) with full Hessian-based GPTQ. Collects X^T X per layer from calibration data, then quantizes weights column-by-column with error compensation into remaining columns. Zero training cost — runs post-training.

## Changes from Base

- Add calibration data collection (128 sequences through trained model)
- Replace quantize_int6_per_row() with full GPTQ column-wise quantization
- Error compensation: W[:, j+1:] -= err * H[j, j+1:] / (H[j,j] + damp)

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
