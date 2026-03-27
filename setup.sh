#!/bin/bash
# Setup for RunPod "Parameter Golf" template (Python 3.12, PyTorch 2.9.1, CUDA 12.8)
# All core deps (torch, sentencepiece, flash_attn) are pre-installed in the template.
set -e

echo "============================================"
echo " Parameter Golf -- Environment Setup"
echo "============================================"

ROOT="$(cd "$(dirname "$0")" && pwd)"

# 1. Extra dependencies (not in base template)
echo ""
echo "[1/2] Extra dependencies..."
pip install -q wandb huggingface-hub zstandard
echo "  Installed."

# 2. Datasets
echo ""
echo "[2/2] FineWeb datasets..."

# sp1024 (used by experiments 01, 03, 05, 06, 07, 08)
echo "  Downloading sp1024..."
cd "$ROOT" && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# sp8192 (used by experiments 02, 04) — not available in upstream yet
echo "  Downloading sp8192 (optional)..."
cd "$ROOT" && python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80 2>/dev/null || \
  echo "  sp8192 not available — experiments 02, 04 will be skipped."

echo "  Done."

# Verification
echo ""
echo "============================================"
echo " Verification"
echo "============================================"

python3 - << 'PYEOF'
import sys, glob
import torch
import numpy as np

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}      : {p.name} ({p.total_mem // 1024**3}GB)")

try:
    import flash_attn_interface
    print(f"FlashAttn3   : available")
except ImportError:
    try:
        import flash_attn
        print(f"FlashAttn    : {flash_attn.__version__}")
    except ImportError:
        print(f"FlashAttn    : NOT found")

for variant in ["sp1024", "sp8192"]:
    train = sorted(glob.glob(f"./data/datasets/fineweb10B_{variant}/fineweb_train_*.bin"))
    val = sorted(glob.glob(f"./data/datasets/fineweb10B_{variant}/fineweb_val_*.bin"))
    if train or val:
        print(f"{variant:12s} : {len(train)} train, {len(val)} val shards")
    else:
        print(f"{variant:12s} : not available")
PYEOF

echo ""
echo "============================================"
echo " Setup complete. Usage:"
echo "   SEED=1337 bash experiments/01_combined_all/run.sh"
echo "   # Or run all: SEED=1337 bash experiments/full_run.sh"
echo "============================================"
