<div align="center">

# HyRES: Residual-Enhanced Hybrid Image Compression

**Minh Khang Tran, Dinh Hoang Dai** &middot; Course research project, 2025

[![Project page](https://img.shields.io/badge/Project-Page-2a78d6?logo=github)](https://tmkhang1999.github.io/HyRES-Residual-Enhanced-Hybrid-Image-Compression/)
[![License](https://img.shields.io/badge/License-Apache_2.0-555555)](LICENSE)
[![Built on CompressAI](https://img.shields.io/badge/Built_on-CompressAI-555555)](https://github.com/InterDigitalInc/CompressAI)

<img src="docs/static/fig/hyres_pipeline.svg" alt="HyRES pipeline: a quality-1 JPEG plus an entropy-coded residual" width="720">

</div>

**Figure 1.** HyRES sends a quality-1 JPEG plus a learned code for the residual *r* = *x* - *x*<sub>J</sub> (what the JPEG got wrong). The decoder adds the decoded residual back. Thumbnails are real outputs of the released Phase 1 model; bitrates are Kodak averages.

## Overview

Standard codecs such as JPEG are fast and supported everywhere but lose detail and color at low bitrates. Learned codecs are far more efficient but large, slow and incompatible with existing viewers. HyRES keeps a standard JPEG as the base layer and spends a small learned codec (10.14M parameters) only on the residual, so the file still contains a valid JPEG. On Kodak, the released model reaches **37.41 dB at 1.563 bpp, 2.1 dB above JPEG at the same bitrate**, and encodes plus decodes an image in about 0.76 s on one GPU. It does not match state-of-the-art learned codecs: ELIC reaches the same quality at about half the bits. HyRES trades compression efficiency for a small model, fast coding and backward compatibility.

<details>
<summary><b>Terms used below</b></summary>

| Term | Meaning |
|------|---------|
| bpp | Bits per pixel: file size in bits divided by the number of pixels. Lower is smaller. |
| PSNR | Peak signal-to-noise ratio in dB, from the mean squared error on the 0-255 scale. Higher is better. |
| MS-SSIM | Multi-scale structural similarity, 0 to 1. Higher is better. |
| Rate-distortion (RD) | The trade-off between file size (rate) and error (distortion). A better codec has a higher RD curve. |
| lambda | Training weight on distortion in `loss = bpp + lambda * MSE`. Higher lambda gives larger files with less error; each lambda gives one model, one point on the RD curve. |
| Phase *k* | The model trained with the *k*-th lambda of the schedule 0.045, 0.032, 0.016, 0.008, 0.004, 0.002. |
| Entropy coding | Lossless compression of quantized values with an arithmetic coder (AE encodes, AD decodes), using predicted probabilities. Better predictions mean fewer bits. |
| Hyperprior | A small side network that sends a few extra bits (*z*) describing how to predict the main latent (*y*). |

</details>

## Method

The residual codec is a scale-hyperprior autoencoder ([Balle et al., ICLR 2018](https://arxiv.org/abs/1802.01436)) with attention modules and residual bottleneck blocks ([Cheng et al., CVPR 2020](https://arxiv.org/abs/2001.01568)) and a checkerboard context model that decodes the latent in two parallel passes instead of one slow autoregressive pass ([He et al., CVPR 2021](https://arxiv.org/abs/2103.15306)).

<p align="center">
  <img src="docs/static/fig/hyres_codec.svg" alt="Residual codec: hyperprior autoencoder with checkerboard context model" width="620">
</p>

**Figure 2.** Residual codec. Blue: learned modules. Teal: arithmetic coding (AE) and decoding (AD) of the quantized latents. The entropy parameters (mean and scale of each latent) come from the hyperprior and, for the second pass, from already decoded anchor positions. Layer lists match `models/checkerboard.py`.

**Training.** `loss = bpp + lambda * MSE` on Mini-ImageNet (60k images, 7 GB), 256x256 crops, one NVIDIA A40. The model is trained in six phases: a high lambda first so it learns to reconstruct, then lambda lowered phase by phase, each phase starting from the previous best checkpoint.

## Results

All results on Kodak (24 images, 768x512), PSNR on RGB, averaged per image.

<p align="center">
  <img src="docs/static/fig/rd_kodak.svg" alt="PSNR versus bitrate on Kodak" width="560">
</p>

**Figure 3.** Rate-distortion on Kodak. The star is the only HyRES result measured from a real bitstream; hollow circles are training-time estimates for the six phases (see [Reproducibility notes](#reproducibility-notes)). JPEG is measured with the same TurboJPEG settings as the HyRES base layer. Published codecs come from [CompressAI's benchmark results](https://github.com/InterDigitalInc/CompressAI/tree/master/results/image/kodak). The interactive version is on the [project page](https://tmkhang1999.github.io/HyRES-Residual-Enhanced-Hybrid-Image-Compression/#results).

| Method | bpp | PSNR (dB) | MS-SSIM |
|--------|----:|----------:|--------:|
| JPEG quality 1 (the HyRES base layer alone) | 0.203 | 21.54 | - |
| JPEG at the same bitrate as HyRES (interpolated from its RD curve) | 1.563 | 35.32 | - |
| **HyRES Phase 1 (lambda = 0.045)** | **1.563** | **37.41** | **0.989** |

HyRES bitrate split: JPEG 0.203 + residual latent *y* 1.332 + hyper-latent *z* 0.028 bpp. Most bits go to the residual at this operating point, so the gain over JPEG comes from coding those bits more efficiently, not from the JPEG layer.

**Efficiency.** Seconds per Kodak image, all models on the same A40 machine, measured during the project:

| Model | Parameters | Encode (s) | Decode (s) | Total (s) |
|-------|-----------:|-----------:|-----------:|----------:|
| Balle (ICLR18), scale hyperprior | - | 0.22 | 0.24 | 0.46 |
| Minnen (NeurIPS18), joint autoregressive | 14.13M | 2.85 | 3.74 | 6.59 |
| Cheng (CVPR20) | 13.18M | 3.57 | 6.56 | 10.13 |
| ELIC (CVPR22) | 33.79M | 4.31 | 4.54 | 8.85 |
| **HyRES** | **10.14M** | 0.476 | 0.286 | **0.762** |

HyRES encodes slower than it decodes because the JPEG step runs on the CPU.

<p align="center">
  <img src="docs/static/fig/visual_comparison.png" alt="Crops of the original, JPEG at equal bitrate and HyRES on three Kodak images" width="900">
</p>

**Figure 4.** Visual comparison at equal bitrate. JPEG uses the lowest quality whose file is at least as large as the HyRES file for that image. The crop is the most textured region of the original, chosen without looking at either reconstruction. Labels give whole-image values.

## Pretrained models

| Model | lambda | Kodak bpp | Kodak PSNR | Download |
|-------|-------:|----------:|-----------:|----------|
| HyRES Phase 1 (no refinement) | 0.045 | 1.563 | 37.41 dB | Coming soon (GitHub Release) |

Only the Phase 1 weights survive from the project. Training checkpoints must be exported once before use (step 2 below).

## Getting started

Requirements: Python 3.10+ and the native TurboJPEG library. Training needs a CUDA GPU; inference runs on CUDA or CPU (not Apple MPS).

```bash
# Native library (Linux / macOS); on Windows install libjpeg-turbo from https://libjpeg-turbo.org
sudo apt-get install libturbojpeg        # or: brew install jpeg-turbo

python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Compress and decompress your own images** with the Phase 1 checkpoint (height and width must be multiples of 32):

```bash
# 1. Export once: rebuilds the entropy-coder tables needed for real bitstreams
bash scripts/export.sh path/to/checkpoint_best_loss_400.pth.tar phase1

# 2. Encode + decode an image or a folder; writes reconstructions and metrics.csv
bash scripts/evaluate.sh checkpoint/inference/phase1.pth.tar my_image.png out/
bash scripts/evaluate.sh checkpoint/inference/phase1.pth.tar             # all of Kodak
```

`metrics.csv` reports the real bitrate (split into JPEG, *y* and *z*), MSE, PSNR, MS-SSIM (if `pytorch-msssim` is installed) and encode/decode time per image.

## Training

Download Mini-ImageNet into `data/train` (needs Kaggle credentials in `~/.kaggle/kaggle.json`, `./kaggle.json`, or `KAGGLE_USERNAME` / `KAGGLE_KEY`; never commit them). `data/test` already holds Kodak.

```bash
pip install kaggle
bash scripts/setup_data.sh
```

Train the six phases; replace `<E>` with the epoch of the best checkpoint in the previous phase's folder:

```bash
bash scripts/train.sh 0.045 checkpoint/phase1
bash scripts/train.sh 0.032 checkpoint/phase2 checkpoint/phase1/checkpoint_best_loss_<E>.pth.tar
bash scripts/train.sh 0.016 checkpoint/phase3 checkpoint/phase2/checkpoint_best_loss_<E>.pth.tar
bash scripts/train.sh 0.008 checkpoint/phase4 checkpoint/phase3/checkpoint_best_loss_<E>.pth.tar
bash scripts/train.sh 0.004 checkpoint/phase5 checkpoint/phase4/checkpoint_best_loss_<E>.pth.tar
bash scripts/train.sh 0.002 checkpoint/phase6 checkpoint/phase5/checkpoint_best_loss_<E>.pth.tar
```

Defaults: batch 16 with 2 gradient-accumulation steps, 256x256 patches, mixed precision, `EPOCHS=1000`, `LR=3e-4` (override with environment variables). The first phase uses noise-based quantization for its early epochs, then straight-through rounding; later phases use straight-through rounding and `ReduceLROnPlateau`. TensorBoard logs go to each save folder (`tensorboard --logdir checkpoint`). Each script prints its usage when called without arguments.

<details>
<summary><b>Optional refinement network</b></summary>

`MultiScaleRefine` (`models/layers/enhancement.py`) is a small post-filter on the decoded image: a squeeze-and-excitation block, dilated convolutions at full, 1/2 and 1/4 resolution, spatial attention and a fusion convolution, added back to the reconstruction. It is trained on a frozen final-phase model:

```bash
bash scripts/train_refine.sh checkpoint/phase6/checkpoint_best_loss_<E>.pth.tar
```

No refinement weights survive, so none of the verified results use it. When a checkpoint has no refinement weights, the model skips refinement and prints a warning.

</details>

## Reproducibility notes

<details>
<summary><b>Numbers reported during the project (all six phases)</b></summary>

These come from the final presentation and cannot be re-measured, because only the Phase 1 weights survive. They were computed during training on center crops, from the model's rate estimate rather than real bitstreams, and that estimate had a bug (see below). For Phase 1 the estimate gave 1.465 bpp against 1.563 bpp measured, so treat the bpp column as indicative only. MSE is not affected by the bug.

| Phase | lambda | Best epoch | bpp (estimated) | MSE (0-255) |
|-------|-------:|-----------:|------:|------:|
| JPEG quality 1 | - | - | 0.2028 | 476.66 |
| 1 | 0.045 | 400 | 1.465 | 11.76 |
| 2 | 0.032 | 41 | 1.109 | 15.72 |
| 3 | 0.016 | 6 | 0.852 | 22.89 |
| 4 | 0.008 | 51 | 0.604 | 41.28 |
| 5 | 0.004 | 13 | 0.460 | 63.79 |
| 6 | 0.002 | 17 | 0.380 | 91.61 |
| 6 + refinement | 0.002 | - | 0.380 | 84.69 (evaluation set not stated) |

</details>

<details>
<summary><b>Codec bugs found when re-measuring, now fixed</b></summary>

Decoding real bitstreams with the Phase 1 weights exposed three bugs in `models/checkerboard.py`. None requires retraining to use the current weights.

1. **Decoded residual was clamped to [0, 1].** Residuals are signed, so every negative correction was lost: 24.46 dB instead of 37.41 dB.
2. **Each checkerboard pass entropy-coded the full tensor**, spending bits on the other pass's empty positions: 1.878 bpp instead of 1.563 bpp.
3. **The training rate estimate scored each latent with the sum of both passes' entropy parameters**, a distribution no decoder has, so it under-estimated the rate (1.279 vs 1.563 bpp on full images). It now uses each position's own parameters (1.666 bpp, a slight over-estimate). Models trained before this fix optimized the wrong rate term, so retraining should improve the RD trade-off.

</details>

<details>
<summary><b>Rebuilding the figures and the project page</b></summary>

Needs the exported Phase 1 model and the saved per-phase reconstructions in `checkpoint/1B`:

```bash
bash scripts/evaluate.sh checkpoint/inference/phase1_lambda0.045.pth.tar data/test output/phase1
python tools/figures/jpeg_curve.py                                    # JPEG RD curve, HyRES settings
python tools/figures/make_figures.py --eval-dir output/phase1 --phase-dir checkpoint/1B   # page images + data
python tools/figures/make_diagrams.py --eval-dir output/phase1        # Figures 1 and 2 (SVG)
python tools/figures/make_plots.py --eval-dir output/phase1           # Figures 3 and 4 (SVG/PDF/PNG)
python -m http.server 8000 --directory docs                           # preview the page
```

PDF versions of Figures 3 and 4 are written next to the SVG/PNG files for use in LaTeX. Baseline curves in `tools/figures/data/compressai_kodak/` are from CompressAI (BSD 3-Clause Clear, license included). To publish the page, enable GitHub Pages for the `docs/` folder of the default branch.

</details>

## Limitations and future work

- **One verified operating point.** A full RD curve and a BD-rate against VVC need several models trained at different lambdas, ideally with the fixed rate estimate.
- **Kodak is not fully held out.** It was also used to pick checkpoints during training; a separate validation set would make the numbers stricter.
- **MSE-only training.** Reconstructions are smooth rather than perceptually sharp. Planned: VGG or GAN losses (`loss = bpp + lambda * MSE + theta * VGG`), multi-scale and wavelet-based blocks against blocking, and a larger training set.
- **Input size.** Height and width must be multiples of 32 (the hyper-latent *z* is at 1/32 resolution); pad other images first.

<details>
<summary><b>Repository layout</b></summary>

```
models/
  hyres.py                 ResidualJPEGCompression: JPEG + residual codec + optional refinement
  checkerboard.py          LightWeightCheckerboard: the residual codec (Figure 2)
  elic.py, cheng2020.py    Reference baselines (not used by HyRES training or inference)
  layers/                  attention, checkerboard masked conv, MultiScaleRefine
  utils/                   JPEG wrappers (TurboJPEG, PIL) and quantizer
src/
  training.py              Train one lambda phase
  refine_training.py       Train the refinement network (base model frozen)
  updata.py                Export a checkpoint with rebuilt entropy-coder tables
  inference.py             Real compress/decompress + metrics
  refine_inference.py      Inference with a separately trained refinement checkpoint
  losses/                  Rate-distortion loss, VGG perceptual loss
  utils/                   Datasets, training loops, optimizers, checkpoints
scripts/                   One shell entry point per step: setup_data, train, train_refine, export, evaluate
tools/figures/             Scripts that build every figure and the project page data
docs/                      Project page (GitHub Pages); static/fig/ holds Figures 1-4
data/test/                 Kodak images (24 PNGs)
data/train/                Mini-ImageNet (downloaded, not tracked)
checkpoint/                Training outputs (not tracked)
```

</details>

## Citation

No paper has been published. If you use this code, please cite the repository:

```bibtex
@misc{hyres2025,
  title        = {HyRES: Residual-Enhanced Hybrid Image Compression},
  author       = {Tran, Minh Khang and Dinh, Hoang Dai},
  year         = {2025},
  howpublished = {\url{https://github.com/tmkhang1999/HyRES-Residual-Enhanced-Hybrid-Image-Compression}}
}
```

## Acknowledgements

Built on [CompressAI](https://github.com/InterDigitalInc/CompressAI). Kodak images by Eastman Kodak, released for unrestricted use. Licensed under Apache 2.0, see [LICENSE](LICENSE).
