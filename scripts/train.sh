#!/usr/bin/env bash
# Train one lambda phase of HyRES.
#
# Usage: scripts/train.sh LAMBDA SAVEPATH [PREV_CHECKPOINT]
#   Phase 1: scripts/train.sh 0.045 checkpoint/phase1
#   Phase 2: scripts/train.sh 0.032 checkpoint/phase2 checkpoint/phase1/checkpoint_best_loss_<E>.pth.tar
# Each later phase starts from the best checkpoint of the previous one.
# Optional env vars: EPOCHS (default 1000), LR (default 3e-4).
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ "$#" -lt 2 ]; then
    sed -n '2,8p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
LAMBDA="$1"
SAVEPATH="$2"
PREV_CKPT="${3:-}"

EXTRA=()
if [ -n "$PREV_CKPT" ]; then
    [ -f "$PREV_CKPT" ] || { echo "error: checkpoint not found: $PREV_CKPT" >&2; exit 1; }
    # --pretrained resets the epoch counter and switches to STE rounding + ReduceLROnPlateau
    EXTRA+=(--pretrained --checkpoint "$PREV_CKPT")
fi

python -m src.training \
    -d ./data \
    --N 128 --M 192 \
    --jpeg-quality 1 \
    --patch-size 256 256 \
    --batch-size 16 --test-batch-size 16 \
    -e "${EPOCHS:-1000}" \
    -lr "${LR:-3e-4}" --aux-learning-rate "${LR:-3e-4}" \
    -n 4 \
    --lambda "$LAMBDA" \
    --alpha 0 \
    --cuda True \
    --save \
    --seed 1926 \
    --clip_max_norm 1.0 \
    --mixed-precision \
    --gradient-accumulation-steps 2 \
    --savepath "$SAVEPATH" \
    ${EXTRA[@]+"${EXTRA[@]}"}
