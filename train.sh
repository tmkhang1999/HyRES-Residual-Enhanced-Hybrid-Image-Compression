python -m src.training \
  -d ./data \
  --N 128 \
  --M 192 \
  --jpeg-quality 1 \
  --patch-size 256 256 \
  --batch-size 16 \
  --test-batch-size 16 \
  -e 1000 \
  -lr 1e-3 \
  --aux-learning-rate 1e-3 \
  -n 4 \
  --lambda 0.032 \
  --cuda True\
  --save \
  --seed 1926 \
  --clip_max_norm 1.0 \
  --mixed-precision \
  --gradient-accumulation-steps 2 \
  --savepath ./checkpoint/phase1

#  --pretrained \
#  --checkpoint Pretrained4000epoch_checkpoint.pth.tar

## Phase 1: mae_loss
# 3e-4
# 0.032

## Phase 2: bpp_loss (reduce 0.19 + 0.2 -> 0.4)
# 1e-3
# 0.016


