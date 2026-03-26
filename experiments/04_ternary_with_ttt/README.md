# 04 Ternary with TTT

**Base**: Ternary submission (73.7M params, 1.1570 BPB)
**Target**: Uncertain — 0.002-0.005 BPB if TTT survives ternary rounding

## Technique

Add Legal Score-First TTT to the ternary (BitNet b1.58) paradigm. Ternary packs 73.7M params into 16MB at ~1.6 bits/param. TTT adapts float weights per-chunk, but forward pass re-quantizes to ternary — the question is whether small TTT updates survive ternary rounding.

## Changes from Base

- Add eval_val_sliding_ttt() from SOTA #1
- SGD(lr=0.002, momentum=0.9), 3 epochs per chunk, all blocks unfrozen
- Score under inference_mode before training on each chunk

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
