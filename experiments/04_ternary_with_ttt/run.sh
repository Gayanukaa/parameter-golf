#!/bin/bash
set -euo pipefail

EXP_NAME="04_ternary_with_ttt"
SEED=${SEED:-1337}
NGPU=${NGPU:-8}
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

# === TRAINING (outputs go to this experiment's folder) ===
cd "$DIR"
RUN_ID="${EXP_NAME}_seed${SEED}" \
DATA_PATH="$ROOT/data/datasets/fineweb10B_sp8192" \
TOKENIZER_PATH="$ROOT/data/tokenizers/fineweb_8192_bpe.model" \
VOCAB_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=600 \
SEED="$SEED" \
NUM_LAYERS=10 \
MODEL_DIM=768 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=4 \
EMBED_DIM=254 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
ROPE_BASE=5000 \
ROPE_TYPE=yarn \
QUANT_TYPE=ternary \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=0 \
EVAL_STRIDE=16 \
torchrun --standalone --nproc_per_node="$NGPU" "$DIR/train_gpt.py"

# === POST-TRAINING ===
python "$ROOT/experiments/shared/metrics.py" \
  --log "$DIR/logs/${EXP_NAME}_seed${SEED}.log" \
  --experiment "$EXP_NAME" --seed "$SEED" \
  --output "$ROOT/experiments/metrics/" \
  --techniques ternary bitnet ttt yarn fp8

if [ "${WANDB_ENABLED:-0}" = "1" ]; then
  python "$ROOT/experiments/shared/wandb_report.py" \
    --metrics "$(ls -t "$ROOT/experiments/metrics/${EXP_NAME}_seed${SEED}"*.json | head -1)" \
    --artifact "$DIR"/final_model.*.ptz || echo "WARNING: wandb upload failed"
fi

if [ "${HF_UPLOAD:-0}" = "1" ]; then
  python "$ROOT/experiments/shared/hf_upload.py" \
    --model "$DIR"/final_model.*.ptz \
    --script "$DIR/train_gpt.py" \
    --metrics "$(ls -t "$ROOT/experiments/metrics/${EXP_NAME}_seed${SEED}"*.json | head -1)" \
    --repo "Gayanukaa" || echo "WARNING: HF upload failed"
fi

echo "Done: $EXP_NAME seed=$SEED"
