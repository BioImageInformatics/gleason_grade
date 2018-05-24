#!/bin/bash

set -e

# FOV experiment
batch_sizes=(12 8 6)
img_ratios=(0.25 0.5 1.0)
crop_sizes=(512 512 512)
epochs=(200 200 200)
lrs=(0.001 0.001 0.001)
basedirs=('5x' '10x' '20x')

# cd into the model directory
cd unet_small

for i in `seq 0 2`; do
  python ./train.py \
  --batch_size ${batch_sizes[$i]} \
  --image_ratio ${img_ratios[$i]} \
  --crop_size ${crop_sizes[$i]} \
  --n_epochs ${epochs[$i]} \
  --lr ${lrs[$i]} \
  --basedir ${basedirs[$i]}
done
