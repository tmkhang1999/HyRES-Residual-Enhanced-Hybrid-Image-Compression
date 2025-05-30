python -m src.training \
  -d ./data \
  --N 128 \
  --M 192 \
  --jpeg-quality 1 \
  --patch-size 256 256 \
  --batch-size 16 \
  --test-batch-size 16 \
  -e 1000 \
  -lr 3e-4 \
  --aux-learning-rate 3e-4 \
  -n 4 \
  --lambda 0.032 \
  --alpha 0\
  --cuda True\
  --save \
  --seed 1926 \
  --clip_max_norm 1.0 \
  --mixed-precision \
  --gradient-accumulation-steps 2 \
  --pretrained \
  --checkpoint ./checkpoint/checkpoint_base.pth.tar \
  --savepath ./checkpoint/phase1 \
