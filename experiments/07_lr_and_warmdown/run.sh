#!/bin/bash
set -euo pipefail

EXP_NAME="07_lr_and_warmdown"
SEED=${SEED:-1337}
NGPU=${NGPU:-8}
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

# Schedule variant: cosine (default), linear, or current baseline
LR_SCHEDULE=${LR_SCHEDULE:-cosine}

# === TRAINING ===
RUN_ID="${EXP_NAME}_${LR_SCHEDULE}_seed${SEED}" \
DATA_PATH="$ROOT/data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="$ROOT/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
SEED="$SEED" \
LR_SCHEDULE="$LR_SCHEDULE" \
WARMDOWN_ITERS=4500 \
NOISY_QAT=1 \
torchrun --standalone --nproc_per_node="$NGPU" "$DIR/train_gpt.py"

# === POST-TRAINING ===
python "$ROOT/experiments/shared/metrics.py" \
  --log "logs/${EXP_NAME}_${LR_SCHEDULE}_seed${SEED}.txt" \
  --experiment "${EXP_NAME}_${LR_SCHEDULE}" --seed "$SEED" \
  --output "$ROOT/experiments/metrics/" \
  --techniques "${LR_SCHEDULE}_schedule" longer_warmdown noisy_qat

[ "${WANDB_ENABLED:-0}" = "1" ] && \
  python "$ROOT/experiments/shared/wandb_report.py" \
    --metrics "$(ls -t "$ROOT/experiments/metrics/${EXP_NAME}_${LR_SCHEDULE}_seed${SEED}"*.json | head -1)" \
    --artifact final_model.*.ptz

[ "${HF_UPLOAD:-0}" = "1" ] && \
  python "$ROOT/experiments/shared/hf_upload.py" \
    --model final_model.*.ptz \
    --script "$DIR/train_gpt.py" \
    --metrics "$(ls -t "$ROOT/experiments/metrics/${EXP_NAME}_${LR_SCHEDULE}_seed${SEED}"*.json | head -1)" \
    --repo "Gayanukaa/parameter-golf-${EXP_NAME}"

echo "Done: $EXP_NAME schedule=$LR_SCHEDULE seed=$SEED"
