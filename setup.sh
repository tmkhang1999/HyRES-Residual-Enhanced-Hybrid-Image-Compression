#!/bin/bash

## This script sets up the environment
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Create and activate a conda environment (recommended for Python 3.12+)
conda create -y -n myenv python=3.12
conda activate myenv

# Upgrade pip (optional with conda, but can still be done)
python -m pip install --upgrade pip

# Install the required Python packages
pip install -r requirements.txt

## Download and prepare the mini-imagenet dataset
# Install kaggle CLI if not already installed
pip install --user kaggle

# Ensure kaggle directory exists
mkdir -p ~/.kaggle

# Note: User needs to place their kaggle.json in the current directory before running this script
# Move kaggle credentials to the right location if it exists in current directory
if [ -f "kaggle.json" ]; then
    mv kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
else
    echo "Please place your kaggle.json credentials file in the current directory"
    exit 1
fi

# Download mini-imagenet dataset from kaggle
kaggle datasets download -d arjunashok33/miniimagenet

# Create directories for ImageFolder structure
mkdir -p data/train data/test

# Unzip the dataset
echo "Extracting dataset..."
unzip miniimagenet.zip -d ./data/train
rm -rf miniimagenet.zip

# Reorganize the dataset into train/test folders
python ./data/reorganize.py