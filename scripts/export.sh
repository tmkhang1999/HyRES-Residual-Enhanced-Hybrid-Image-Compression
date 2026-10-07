#!/usr/bin/env bash
# Export a training checkpoint for inference. This rebuilds the entropy-coder
# CDF tables, which compress()/decompress() need to produce a real bitstream.
#
# Usage: scripts/export.sh CHECKPOINT NAME
#   scripts/export.sh checkpoint/phase1/checkpoint_best_loss_98.pth.tar phase1_0.032
# Output: checkpoint/inference/NAME.pth.tar
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ "$#" -lt 2 ]; then
    sed -n '2,7p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
[ -f "$1" ] || { echo "error: checkpoint not found: $1" >&2; exit 1; }

python -m src.updata \
    --filepath "$1" \
    --name "$2" \
    --dir ./checkpoint/inference \
    --N 128 --M 192 \
    --jpeg-quality 1
