#!/bin/bash

set -e

# batch_sizes=(32 16 4)
# img_ratios=(0.25 0.5 1.0)
# basedirs=('5x' '10x' '20x')

batch_sizes=(32 16 4)
img_ratios=(0.25 0.5 1.0)
basedirs=('5x' '10x' '20x')

for i in `seq 0 2`; do
  for k in `seq 0 4`; do
    python ./train.py ${batch_sizes[$i]} ${img_ratios[$i]} ${basedirs[$i]}
  done
done
