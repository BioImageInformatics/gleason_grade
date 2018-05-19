#!/bin/bash

set -e

# FOV experiment
batch_sizes=(24 24 24)
img_ratios=(0.25 0.5 1.0)
crop_sizes=(1024 512 256)
epochs=(200 400 750)
lrs=(0.0001 0.0001 0.0001)
basedirs=('5x_FOV' '10x_FOV' '20x_FOV')

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
