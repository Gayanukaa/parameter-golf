# 08 SSM Hybrid

**Base**: Best from above experiments
**Target**: Uncertain — on OpenAI's wishlist, high visibility even as non-record

## Technique

Replace middle 3-4 transformer layers with Mamba/S4 state-space model blocks. SSM is O(n) in sequence length vs O(n^2) for attention, allowing either more layers or longer sequences within the same compute budget. Transformer layers kept at bottom (embedding extraction) and top (prediction).

## Changes from Base

- Replace layers 4-7 with minimal S4 blocks (conv1d + selective scan)
- Keep attention layers at bottom and top of stack
- May need mamba-ssm pip package for optimized CUDA kernels

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
