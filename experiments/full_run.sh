#!/bin/bash
set -euo pipefail

SEED=${SEED:-1337}
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

# Experiments requiring sp8192 dataset (02, 04) are skipped if not available
SP8192_AVAILABLE=0
[ -d "$ROOT/data/datasets/fineweb10B_sp8192" ] && SP8192_AVAILABLE=1

EXPERIMENTS=(
  01_combined_all
  03_full_gptq
  05_float8_training
  06_deeper_model
  07_lr_and_warmdown
  08_ssm_hybrid
)

# Add sp8192 experiments if dataset is available
if [ "$SP8192_AVAILABLE" = "1" ]; then
  EXPERIMENTS=(01_combined_all 02_8192_vocab 03_full_gptq 04_ternary_with_ttt 05_float8_training 06_deeper_model 07_lr_and_warmdown 08_ssm_hybrid)
else
  echo "Note: sp8192 dataset not found — skipping experiments 02, 04"
fi

# Allow selecting specific experiments: EXPS="01 03 07" bash full_run.sh
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
