# 02 8192 BPE Vocab

**Base**: Best from 01_combined_all
**Target**: 0.005-0.015 BPB improvement via better bytes-per-token ratio

## Technique

Switch from 1024 to 8192 BPE vocabulary. Larger vocab represents more bytes per token, directly improving BPB since evaluation is tokenizer-agnostic. Uses factored tied embeddings (8192->254->model_dim) to keep artifact size under 16MB.

## Changes from Base

- VOCAB_SIZE=8192 with sp8192 tokenizer and data
- Factored tied embeddings: tok_emb(8192, 254) + projection layers
- Adjusted embedding LR for larger vocab

## Prereq

```bash
python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

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
