# HyRES: Residual-Enhanced Hybrid Image Compression

HyRES is a state-of-the-art hybrid image compression framework that combines traditional JPEG compression with neural network-based residual compression. This approach achieves superior compression efficiency while maintaining high image quality.

## Architecture Overview

The framework consists of three main components:

1. **JPEG Compression**: Traditional JPEG compression with configurable quality factor
2. **Residual Compression**: Neural network-based compression of the JPEG residual
3. **Multi-Scale Refinement**: Post-processing network to enhance the final reconstruction

### Key Features

- **Hybrid Compression**: Combines JPEG and neural compression for optimal rate-distortion performance
- **Residual Learning**: Focuses on compressing the difference between original and JPEG-compressed images
- **Multi-Scale Enhancement**: Uses a multi-scale refinement network to improve visual quality
- **Attention Mechanisms**: Incorporates both channel and spatial attention for better feature extraction
- **Checkerboard Pattern**: Implements a checkerboard pattern for efficient context modeling

## Project Structure

```
HyRES-Residual-Enhanced-Hybrid-Image-Compression
├── checkpoints/                # Model checkpoints directory
├── data/                      # Dataset directory
│   ├── train/                 # Training images
│   └── test/                  # Test images
├── models/                    # Model architecture definitions
│   ├── hyres.py               # ResidualJPEGCompression (HyRES main model)
│   ├── checkerboard.py        # LightWeightCheckerboard (residual model)
│   ├── layers/                # Custom neural network layers
│   │   ├── attention.py       # AttentionBlock
│   │   ├── checkerboard.py    # MaskedConv2d, CheckboardMaskedConv2d
│   │   ├── common.py          # conv1x1, conv3x3 helpers
│   │   └── enhancement.py     # MultiScaleRefine (refinement network)
│   └── utils/                 # Model utilities
│       ├── jpeg_compression.py  # JPEG compression/decompression module
│       └── quantization.py      # Quantizer class with noise and STE support
├── src/                      # Source code
│   ├── losses/               # Loss functions
│   │   ├── rd_loss.py        # RateDistortionLoss implementation
│   │   └── vgg16.py          # VGG-based perceptual loss
│   ├── utils/                # Training utilities
│   │   ├── checkpoint_utils.py  # Checkpoint management functions
│   │   ├── dataset_utils.py     # ImageFolder dataset implementation
│   │   ├── engine.py            # Training and testing loops
│   │   └── optimizers.py        # Optimizer configuration
│   ├── inference.py          # Model inference script
│   ├── training.py           # Main training script
│   ├── refine_training.py    # Refinement network training script
│   ├── refine_inference.py   # Refinement network inference script
│   └── updata.py             # Model update utilities
├── .gitignore               # Git ignore file
├── LICENSE                  # Apache-2.0 license file
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── setup.sh                 # Environment setup script
├── train.sh                 # Training script
└── test.sh                  # Model testing script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/HyRES-Residual-Enhanced-Hybrid-Image-Compression.git
cd HyRES-Residual-Enhanced-Hybrid-Image-Compression
```

2. Set up the environment:
```bash
chmod +x setup.sh
./setup.sh
```

## Usage

### Training

To train the model:

```bash
chmod +x train.sh
./train.sh
```

Key training parameters:
- `--dataset`: Path to training dataset
- `--N`: Number of channels in main codec (default: 128)
- `--M`: Number of channels in latent space (default: 192)
- `--jpeg-quality`: JPEG quality factor (default: 1)
- `--epochs`: Number of training epochs (default: 4000)
- `--learning-rate`: Learning rate (default: 1e-4)

### Inference

To compress and decompress images:

```bash
python src/inference.py \
    --input path/to/input/image.png \
    --output path/to/output/image.png \
    --model-checkpoint path/to/checkpoint.pth \
    --jpeg-quality 50
```

## Model Architecture

### ResidualJPEGCompression

The main model combines:
- JPEG compression/decompression
- Neural residual compression
- Multi-scale refinement network

### LightWeightCheckerboard

The residual compression model features:
- Analysis and synthesis transforms
- Hyperprior network
- Checkerboard pattern for context modeling
- Attention blocks for feature enhancement

## Loss Functions

The model uses a combination of:
- Rate-distortion loss
- MSE loss
- VGG perceptual loss

## Multi-Phase Training Strategy

We employ a multi-phase training approach to jointly optimize for both distortion (MSE loss) and rate (bits-per-pixel, bpp):

- **Phase 1:** Start with a high Lagrangian multiplier (λ = 0.045), focusing primarily on image quality. At this stage, the task is treated more as image reconstruction than compression.
- **Subsequent Phases:** Gradually decrease λ to shift the optimization focus toward the rate-distortion trade-off. The λ schedule is as follows:
  - λ: 0.045 → 0.032 → 0.016 → 0.008 → 0.004 → 0.002

This staged reduction in λ allows the model to first learn to reconstruct images well, then progressively balance between image quality and compression rate.

## Refinement Training

For the refinement phase:
- We utilize the pretrained checkpoint obtained from the final phase (phase 6, λ = 0.002).
- The main model weights are **frozen** during refinement.
- Only the refinement network (e.g., multi-scale enhancement or post-processing blocks) is trained, further improving the perceptual quality of the reconstructed images.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hyres2024,
  title={HyRES: Residual-Enhanced Hybrid Image Compression},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```angular2html
HyRES-Residual-Enhanced-Hybrid-Image-Compression
├── checkpoints
├── data
│   ├── train
│   └── test
├── models
│   ├── hyres.py
│   ├── checkerboard.py
│   ├── layers
│   │   ├── attention.py
│   │   ├── checkerboard.py
│   │   ├── common.py
│   │   └── enhancement.py
│   └── utils/
│       ├── jpeg_compression.py
│       └── quantization.py
├── src/
│   ├── losses/
│   │   ├── rd_loss.py
│   │   └── vgg16.py
│   ├── utils/
│   │   ├── checkpoint_utils.py
│   │   ├── dataset_utils.py
│   │   ├── engine.py
│   │   └── optimizers.py
│   ├── inference.py
│   ├── training.py
│   ├── refine_training.py
│   ├── refine_inference.py
│   └── updata.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── setup.sh
└── train.sh
└── test.sh
```