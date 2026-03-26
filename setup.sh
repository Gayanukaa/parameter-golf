#!/bin/bash
set -e

echo "============================================"
echo " Parameter Golf -- Environment Setup"
echo "============================================"

# 1. Miniconda
echo ""
echo "[1/5] Miniconda..."
if [ -d "$HOME/miniconda3" ]; then
    echo "  Already installed -- skipping."
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b
    rm /tmp/miniconda.sh
    ~/miniconda3/bin/conda init bash
    echo "  Installed."
fi

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

echo "  Accepting conda TOS..."
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 2. Python 3.13 environment
echo ""
echo "[2/5] Python 3.13 environment..."
if conda env list | grep -q "^golf "; then
    echo "  Environment 'golf' exists -- activating."
else
    conda create -n golf python=3.13 -y
    echo "  Created."
fi
conda activate golf

# 3. Python dependencies
echo ""
echo "[3/5] Dependencies..."
if python3 -c "import torch, sentencepiece, numpy" 2>/dev/null; then
    echo "  Core packages already installed -- skipping."
else
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    pip install wandb huggingface-hub zstandard -q
    echo "  Installed."
fi

# 4. FlashAttention-3 (H100 optimized)
echo ""
echo "[4/5] FlashAttention-3..."
if python3 -c "import flash_attn" 2>/dev/null || python3 -c "import flash_attn_interface" 2>/dev/null; then
    echo "  Already installed -- skipping."
else
    pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    echo "  Installed."
fi

# 5. Datasets
echo ""
echo "[5/5] FineWeb datasets..."

ROOT="$(cd "$(dirname "$0")" && pwd)"

# sp1024 (used by experiments 01, 03, 05, 06, 07, 08)
echo "  Downloading sp1024 (80 train shards)..."
cd "$ROOT" && python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# sp8192 (used by experiments 02, 04)
echo "  Downloading sp8192 (80 train shards)..."
cd "$ROOT" && python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80

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
    import flash_attn
    print(f"FlashAttn    : {flash_attn.__version__}")
except ImportError:
    try:
        import flash_attn_interface
        print(f"FlashAttn3   : available")
    except ImportError:
        print(f"FlashAttn    : NOT found")

for variant in ["sp1024", "sp8192"]:
    train = sorted(glob.glob(f"./data/datasets/fineweb10B_{variant}/fineweb_train_*.bin"))
    val = sorted(glob.glob(f"./data/datasets/fineweb10B_{variant}/fineweb_val_*.bin"))
    print(f"{variant:12s} : {len(train)} train, {len(val)} val shards")
PYEOF

echo ""
echo "============================================"
echo " Setup complete. Usage:"
echo "   conda activate golf"
echo "   SEED=1337 bash experiments/01_combined_all/run.sh"
echo "   # Or run all: SEED=1337 bash experiments/full_run.sh"
echo "============================================"
