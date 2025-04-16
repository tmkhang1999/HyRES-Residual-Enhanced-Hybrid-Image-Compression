#!/bin/bash

# Install the required Python packages
pip install -r requirements.txt

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
unzip archive.zip -d ./data/train
rm -rf archive.zip

# Move the images to the appropriate directories
python ./data/reorganize.py