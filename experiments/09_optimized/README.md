# 09_optimized

All SOTA techniques from 01_combined_all, tuned for maximum throughput and calibrated warmdown.

## Changes from 01

- `ITERATIONS=9000` (matches SOTA, tighter LR schedule)
- `WARMDOWN_ITERS=2500` (calibrated for ~5000 actual steps)
- `EMA_DECAY=0.995` (faster averaging for shorter training)
- `LATE_QAT_THRESHOLD=0.20` (start QAT earlier, more quantization-aware steps)
- `VAL_LOSS_EVERY=0` (skip mid-training validation, save time)
- `EMA_DECAY` now configurable via env var

## Run

```bash
SEED=1337 bash experiments/09_optimized/run.sh
```
