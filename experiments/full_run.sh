#!/bin/bash
set -euo pipefail

SEED=${SEED:-1337}
DIR="$(cd "$(dirname "$0")" && pwd)"

EXPERIMENTS=(
  01_combined_all
  03_full_gptq
  05_float8_training
  07_lr_and_warmdown
  09_optimized
)

# Allow selecting specific experiments: EXPS="01 09" bash full_run.sh
if [ -n "${EXPS:-}" ]; then
  FILTERED=()
  for e in "${EXPERIMENTS[@]}"; do
    for p in $EXPS; do
      [[ "$e" == "${p}"* ]] && FILTERED+=("$e")
    done
  done
  EXPERIMENTS=("${FILTERED[@]}")
fi

echo "Running ${#EXPERIMENTS[@]} experiments with seed=$SEED"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "---"

for exp in "${EXPERIMENTS[@]}"; do
  echo "=== Starting: $exp (seed=$SEED) ==="
  SEED="$SEED" \
  WANDB_ENABLED="${WANDB_ENABLED:-0}" \
  HF_UPLOAD="${HF_UPLOAD:-0}" \
  bash "$DIR/$exp/run.sh"
  echo "=== Finished: $exp ==="
  echo ""
done

# Print comparison table
echo "=== Results ==="
python "$DIR/shared/compare.py"
