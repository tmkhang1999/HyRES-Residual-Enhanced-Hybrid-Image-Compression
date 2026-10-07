#!/usr/bin/env bash
# Compress and decompress images with an exported model; prints bpp, MSE, PSNR
# and encode/decode times, and writes metrics.csv.
#
# Usage: scripts/evaluate.sh EXPORTED_MODEL [INPUT] [OUTPUT_DIR]
#   INPUT defaults to ./data/test (Kodak), OUTPUT_DIR to ./output
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ "$#" -lt 1 ]; then
    sed -n '2,6p' "$SELF" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
[ -f "$1" ] || { echo "error: model not found: $1" >&2; exit 1; }

python -m src.inference \
    --checkpoint "$1" \
    --input "${2:-./data/test}" \
    --output "${3:-./output}" \
    --N 128 --M 192 \
    --jpeg-quality 1 \
    --cuda true \
    --save-components
