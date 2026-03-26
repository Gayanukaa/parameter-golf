# 06 Deeper Model

**Base**: Best from above experiments
**Target**: 0.005-0.010 BPB per added layer (diminishing returns)

## Technique

Increase from 11 to 13-14 transformer layers using selective activation checkpointing. Checkpointing attention outputs (most memory-hungry) frees ~50% activation memory at ~20% compute cost, allowing more depth within the same GPU memory.

## Changes from Base

- NUM_LAYERS=14 (up from 11)
- Selective activation checkpointing on attention forward pass
- Verify artifact stays < 16MB (each layer ~1.7MB at int6)

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
