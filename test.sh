python -m src.inference \
  --dataset ./data/test \
  --output_path Reconstructed \
  --cuda \
  --patch 256 \
  --N 128 \
  --M 192 \
  --jpeg-quality 1 \
  --path ./checkpoint/inference/phase1_0.032.pth.tar