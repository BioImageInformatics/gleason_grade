#!/bin/bash

set -e

batch_sizes=(16)
img_ratios=(0.5)
crop_sizes=(512)
epochs=(300)
lrs=(0.001)
basedirs=('10x_LONG')

# cd into the model directory
cd densenet

for i in `seq 0 2`; do
  python ./train.py \
  --batch_size ${batch_sizes[$i]} \
  --image_ratio ${img_ratios[$i]} \
  --crop_size ${crop_sizes[$i]} \
  --n_epochs ${epochs[$i]} \
  --lr ${lrs[$i]} \
  --basedir ${basedirs[$i]}
done
