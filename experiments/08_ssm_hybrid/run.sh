#!/bin/bash
set -euo pipefail

EXP_NAME="08_ssm_hybrid"
SEED=${SEED:-1337}
NGPU=${NGPU:-8}
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

# === TRAINING ===
RUN_ID="${EXP_NAME}_seed${SEED}" \
DATA_PATH="$ROOT/data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="$ROOT/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
SEED="$SEED" \
SSM_LAYERS="4,5,6,7" \
TRAIN_SEQ_LEN=2048 \
torchrun --standalone --nproc_per_node="$NGPU" "$DIR/train_gpt.py"

# === POST-TRAINING ===
python "$ROOT/experiments/shared/metrics.py" \
  --log "logs/${EXP_NAME}_seed${SEED}.txt" \
  --experiment "$EXP_NAME" --seed "$SEED" \
  --output "$ROOT/experiments/metrics/" \
  --techniques ssm_hybrid mamba

[ "${WANDB_ENABLED:-0}" = "1" ] && \
  python "$ROOT/experiments/shared/wandb_report.py" \
    --metrics "$(ls -t "$ROOT/experiments/metrics/${EXP_NAME}_seed${SEED}"*.json | head -1)" \
    --artifact final_model.*.ptz

[ "${HF_UPLOAD:-0}" = "1" ] && \
  python "$ROOT/experiments/shared/hf_upload.py" \
    --model final_model.*.ptz \
    --script "$DIR/train_gpt.py" \
    --metrics "$(ls -t "$ROOT/experiments/metrics/${EXP_NAME}_seed${SEED}"*.json | head -1)" \
    --repo "Gayanukaa/parameter-golf-${EXP_NAME}"

echo "Done: $EXP_NAME seed=$SEED"
