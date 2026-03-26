#!/bin/bash
set -euo pipefail

EXP_NAME="06_deeper_model"
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
NUM_LAYERS=14 \
ACTIVATION_CHECKPOINT=selective \
torchrun --standalone --nproc_per_node="$NGPU" "$DIR/train_gpt.py"

# === POST-TRAINING ===
python "$ROOT/experiments/shared/metrics.py" \
  --log "$DIR/logs/${EXP_NAME}_seed${SEED}.txt" \
  --experiment "$EXP_NAME" --seed "$SEED" \
  --output "$ROOT/experiments/metrics/" \
  --techniques deeper_model activation_checkpoint

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
