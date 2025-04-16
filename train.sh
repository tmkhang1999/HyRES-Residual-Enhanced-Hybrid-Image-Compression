python -m src.training \
  -d ./data \
  --N 128 \
  --M 192 \
  --jpeg-quality 10 \
  --patch-size 256 256 \
  --batch-size 16 \
  --test-batch-size 16 \
  -e 2000 \
  -lr 1e-4 \
  --aux-learning-rate 1e-3 \
  -n 8 \
  --lambda 13e-3 \
  --cuda True\
  --save \
  --seed 1926 \
  --clip_max_norm 1.0 \

#  --pretrained \
#  --checkpoint Pretrained4000epoch_checkpoint.pth.tar



