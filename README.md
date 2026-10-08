<div align="center">

# HyRES: Residual-Enhanced Hybrid Image Compression

**Minh Khang Tran, Dinh Hoang Dai** &middot; Course research project, 2025

[![Project page](https://img.shields.io/badge/Project-Page-2a78d6?logo=github)](https://tmkhang1999.github.io/HyRES-Residual-Enhanced-Hybrid-Image-Compression/)
[![License](https://img.shields.io/badge/License-Apache_2.0-555555)](LICENSE)
[![Built on CompressAI](https://img.shields.io/badge/Built_on-CompressAI-555555)](https://github.com/InterDigitalInc/CompressAI)

<img src="docs/static/fig/hyres_pipeline.svg" alt="HyRES pipeline: a quality-1 JPEG plus an entropy-coded residual" width="900">

</div>

**Figure 1.** HyRES stores a quality-1 JPEG and a learned code for the residual *r* = *x* - *x*<sub>J</sub>, the part the JPEG got wrong. The decoder adds the decoded residual back. Thumbnails and bitrates are real outputs of the released model on Kodak image 23 (residual shown 2x).

## Overview

In this project, we combine a standard JPEG with a small learned codec (10.14M parameters). The first idea is to keep JPEG as the base layer, because it is fast and every viewer can open it. The second idea is to use the learned codec only for the residual, because that is where JPEG loses detail at low bitrates. As a result, every HyRES file still contains a valid JPEG.

On Kodak, the released model reaches 37.41 dB at 1.563 bits per pixel (bpp), which is 2.1 dB above JPEG at the same file size (higher dB is better). It encodes and decodes an image in about 0.76 s on one GPU. However, state-of-the-art learned codecs are more efficient: ELIC needs only 0.83 bpp for the same quality. HyRES therefore trades some compression for a small, fast and backward-compatible codec.

## Method

<p align="center">
  <img src="docs/static/fig/hyres_codec.svg" alt="Residual codec: analysis and synthesis transforms, quantization and arithmetic coding, and the entropy model" width="900">
</p>

**Figure 2.** The residual codec. (a) The analysis transform *g*<sub>a</sub> maps the residual to a latent *y* at 1/8 resolution, which is quantized and arithmetic-coded (AE encodes, AD decodes); the synthesis transform *g*<sub>s</sub> maps it back. (b) The entropy model predicts a mean and scale for every latent from a hyperprior ([Balle et al., 2018](https://arxiv.org/abs/1802.01436)) and a checkerboard context model that decodes in two parallel passes ([He et al., 2021](https://arxiv.org/abs/2103.15306)). Attention and residual bottleneck blocks follow [Cheng et al., 2020](https://arxiv.org/abs/2001.01568).

We train with the loss `bpp + lambda * MSE`, where lambda sets the trade-off between file size and error, on Mini-ImageNet (60k images) with one NVIDIA A40. Training has six phases, one per lambda. Phase 1 uses a high lambda (0.045), so the model first learns to reconstruct; lambda then drops to 0.002 in Phase 6, and each phase starts from the previous best checkpoint. The released model is Phase 1.

## Results

We evaluate on Kodak (24 images, 768x512). All HyRES numbers are measured from real bitstreams.

<p align="center">
  <img src="docs/static/fig/rd_kodak.svg" alt="PSNR versus bitrate on Kodak" width="620">
</p>

**Figure 3.** Rate-distortion on Kodak. The filled blue dot is the released model; the hollow circles are training-time estimates of all six phases, not real bitstreams. Published codecs are from [CompressAI](https://github.com/InterDigitalInc/CompressAI/tree/master/results/image/kodak).

From the table below, the first thing we can see is that JPEG needs 2.17 bpp to reach the quality HyRES reaches at 1.56 bpp (28% fewer bits). Moreover, HyRES is the smallest model and, apart from the 2018 hyperprior, the fastest.

| Method | bpp at 37.4 dB | Params | Encode + decode (s) |
|--------|---------------:|-------:|--------------------:|
| JPEG (TurboJPEG) | 2.17 | - | - |
| **HyRES** | **1.56** | **10.1M** | **0.76** |
| Balle (ICLR18) | 1.06 | - | 0.46 |
| VVC (VTM) | 0.87 | - | - |
| ELIC (CVPR22) | 0.83 | 33.8M | 8.85 |
| Cheng (CVPR20) | curve ends at 36.6 dB | 13.2M | 10.13 |

Times are per Kodak image on one A40. Most of the 1.56 bpp (1.36) goes to the residual, so the gain over JPEG comes from coding those bits more efficiently.

<p align="center">
  <img src="docs/static/fig/visual_comparison.png" alt="Crops of the original, JPEG at equal bitrate and HyRES on three Kodak images" width="900">
</p>

**Figure 4.** Visual comparison at equal bitrate. JPEG gets the lowest quality whose file is at least as large as the HyRES file, and each crop is the most textured region of the original. The [project page](https://tmkhang1999.github.io/HyRES-Residual-Enhanced-Hybrid-Image-Compression/) has an interactive version.

## Pretrained model

| Model | lambda | Kodak bpp | Kodak PSNR | Download |
|-------|-------:|----------:|-----------:|----------|
| HyRES Phase 1 | 0.045 | 1.563 | 37.41 dB | Coming soon (GitHub Release) |

## Quick start

We need Python 3.10+ and the TurboJPEG library (`apt install libturbojpeg`, `brew install jpeg-turbo`, or [libjpeg-turbo](https://libjpeg-turbo.org) on Windows). Inference runs on CUDA or CPU.

```bash
pip install -r requirements.txt

# Export once (builds the entropy-coder tables), then encode and decode
bash scripts/export.sh path/to/checkpoint_best_loss_400.pth.tar phase1
bash scripts/evaluate.sh checkpoint/inference/phase1.pth.tar my_image.png out/
```

`out/` receives the reconstruction and a `metrics.csv` with the real bitrate, PSNR, MS-SSIM and timing. Image sides must be multiples of 32. Without an image argument, `evaluate.sh` runs on all of Kodak.

## Training

```bash
bash scripts/setup_data.sh                       # Mini-ImageNet -> data/train (needs Kaggle credentials)
bash scripts/train.sh 0.045 checkpoint/phase1    # Phase 1, from scratch
i=1
for lmbda in 0.032 0.016 0.008 0.004 0.002; do   # each phase starts from the previous best
  prev=$(ls checkpoint/phase$i/checkpoint_best_loss_*.pth.tar); i=$((i+1))
  bash scripts/train.sh $lmbda checkpoint/phase$i "$prev"
done
```

Every script prints its options when run without arguments. An optional post-filter (`scripts/train_refine.sh`) can be trained on the final phase, but the released model does not use it. The scripts in `tools/figures/` rebuild every figure.

## Limitations

However, we should note three limits. First, only the Phase 1 model is verified from real bitstreams, so a full rate-distortion curve and a BD-rate against VVC need the other lambdas retrained; the project page lists the codec bugs we fixed while re-measuring. Second, Kodak also selected checkpoints during training, so it is not fully held out. Third, we train with MSE only, so reconstructions are smooth rather than sharp. In sum, perceptual losses (VGG, GAN) and a larger training set are the natural next steps.

## Citation

```bibtex
@misc{hyres2025,
  title        = {HyRES: Residual-Enhanced Hybrid Image Compression},
  author       = {Tran, Minh Khang and Dinh, Hoang Dai},
  year         = {2025},
  howpublished = {\url{https://github.com/tmkhang1999/HyRES-Residual-Enhanced-Hybrid-Image-Compression}}
}
```

Built on [CompressAI](https://github.com/InterDigitalInc/CompressAI). Kodak images by Eastman Kodak. Apache 2.0 license.
