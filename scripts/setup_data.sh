#!/usr/bin/env bash
# Download Mini-ImageNet (Kaggle) into data/train and flatten it for ImageFolder.
# Prerequisites: `pip install kaggle` and Kaggle credentials, either in
# ~/.kaggle/kaggle.json, in ./kaggle.json (copied for you), or via the
# KAGGLE_USERNAME / KAGGLE_KEY environment variables.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v kaggle >/dev/null 2>&1; then
    echo "error: the 'kaggle' CLI was not found. Run: pip install kaggle" >&2
    exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_KEY:-}" ]; then
    if [ -f kaggle.json ]; then
        mkdir -p "$HOME/.kaggle"
        cp kaggle.json "$HOME/.kaggle/kaggle.json"
        chmod 600 "$HOME/.kaggle/kaggle.json"
    else
        echo "error: no Kaggle credentials found (see header of this script)." >&2
        exit 1
    fi
fi

mkdir -p data/train data/test
kaggle datasets download -d arjunashok33/miniimagenet -p data
# Python's zipfile is used so this also works where `unzip` is missing (Windows).
python -m zipfile -e data/miniimagenet.zip data/train
rm -f data/miniimagenet.zip

# Move images out of class sub-folders: data/train/<class>/x.JPEG -> data/train/x.JPEG
python data/reorganize.py

# data/test must hold the Kodak images (24 PNGs); they are already tracked in git.
echo "Done. Train images: $(find data/train -type f | wc -l)"
