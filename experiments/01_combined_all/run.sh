#!/bin/bash
set -euo pipefail

EXP_NAME="01_combined_all"
SEED=${SEED:-1337}
NGPU=${NGPU:-8}
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

# === TRAINING (outputs go to this experiment's folder) ===
cd "$DIR"
RUN_ID="${EXP_NAME}_seed${SEED}" \
DATA_PATH="$ROOT/data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="$ROOT/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
SEED="$SEED" \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
WARMDOWN_ITERS=3500 \
MATRIX_LR=0.025 \
TIED_EMBED_LR=0.035 \
SCALAR_LR=0.025 \
MUON_MOMENTUM=0.99 \
MUON_BACKEND_STEPS=5 \
GRAD_CLIP_NORM=0.3 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
EMA_DECAY=0.997 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=0 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node="$NGPU" "$DIR/train_gpt.py"

# === POST-TRAINING ===
python "$ROOT/experiments/shared/metrics.py" \
  --log "$DIR/logs/${EXP_NAME}_seed${SEED}.log" \
  --experiment "$EXP_NAME" --seed "$SEED" \
  --output "$ROOT/experiments/metrics/" \
  --techniques leaky_relu_sq ttt parallel_muon gptq_lite ema xsa partial_rope ln_scale

[ "${WANDB_ENABLED:-0}" = "1" ] && \
  python "$ROOT/experiments/shared/wandb_report.py" \
    --metrics "$(ls -t "$ROOT/experiments/metrics/${EXP_NAME}_seed${SEED}"*.json | head -1)" \
    --artifact "$DIR"/final_model.*.ptz

[ "${HF_UPLOAD:-0}" = "1" ] && \
  python "$ROOT/experiments/shared/hf_upload.py" \
    --model "$DIR"/final_model.*.ptz \
    --script "$DIR/train_gpt.py" \
    --metrics "$(ls -t "$ROOT/experiments/metrics/${EXP_NAME}_seed${SEED}"*.json | head -1)" \
    --repo "Gayanukaa"

echo "Done: $EXP_NAME seed=$SEED"
