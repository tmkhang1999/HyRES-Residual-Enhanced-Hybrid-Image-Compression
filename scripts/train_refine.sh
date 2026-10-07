#!/usr/bin/env bash
# Train the refinement network on top of a frozen HyRES checkpoint.
#
# Usage: scripts/train_refine.sh CHECKPOINT [SAVEPATH]
#   CHECKPOINT is the best checkpoint of the last lambda phase (lambda = 0.002).
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ "$#" -lt 1 ]; then
    sed -n '2,5p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
[ -f "$1" ] || { echo "error: checkpoint not found: $1" >&2; exit 1; }

python -m src.refine_training \
    --dataset ./data \
    --N 128 --M 192 \
    --jpeg-quality 1 \
    --batch-size 16 --test-batch-size 16 \
    --patch-size 256 256 \
    --num-workers 4 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --checkpoint "$1" \
    --savepath "${2:-./checkpoint/refine}" \
    --cuda True \
    --seed 1926
