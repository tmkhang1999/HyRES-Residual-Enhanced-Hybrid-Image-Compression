# HyRES-Residual-Enhanced-Hybrid-Image-Compression

```angular2html
HyRES-Residual-Enhanced-Hybrid-Image-Compression
├── checkpoints
├── data
│   ├── train
│   │   ├── img000.png
│   │   └── img001.png
│   ├── test
│   │   ├── img000.png
│   │   └── img001.png
├── models
│   ├── layers
│   │   ├── __init__.py
│   │   ├── attention.py        # Defines `AttentionBlock`
│   │   ├── checkerboard.py     # Defines `MaskedConv2d` and `CheckboardMaskedConv2d` for masked convolutions
│   │   └── common.py           # Provides helper functions: `conv1x1` and `conv3x3`
│   └── utils/
│       ├── jpeg_compression.py # Defines `JPEGCompression` (nn.Module) for JPEG compression/decompression
│       └── quantization.py     # Contains `Quantizer` class with `quantize()` method (supports noise and STE)
├── src/
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── rd_loss.py          # Implements `RateDistortionLoss` (nn.Module) for RD loss calculation
│   │   └── vgg16.py            # Implements `VGGLoss` (nn.Module) for VGG-based perceptual loss
│   ├── utils/
│   │   ├── checkpoint_utils.py # Functions: `DelfileList`, `load_checkpoint`, and `save_checkpoint`
│   │   ├── dataset_utils.py    # Defines `ImageFolder` (Dataset) for loading images from folders
│   │   ├── engine.py           # Contains `train_one_epoch` and `test_epoch` for training/testing loops
│   │   └── optimizers.py       # Provides `configure_optimizers` to set up main and auxiliary optimizers
│   ├── inference.py            # Script/module for running model inference
│   ├── training.py             # Script/module for model training routines
│   └── updata.py               # Script/module likely for updating model checkpoints or metrics
├── .gitignore                  # Git ignore file
├── LICENSE                     # Apache-2.0 license file
├── README.md                   # Project overview and instructions
├── requirements.txt            # Python dependencies for the project
├── setup.sh                    # Shell script for setting up the environment
└── train.sh                    # Shell script for training the model
```