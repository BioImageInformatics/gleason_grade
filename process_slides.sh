#!/bin/bash

set -e

svsdir=data/validation_svs

outdir=(
<<<<<<< HEAD
densenet/ext_5x/inference
)

snapshot=(
densenet/ext_5x/snapshots/densenet.ckpt-355669
)

mags=( 5 )
sizes=( 256 )
batches=( 8 )
=======
densenet_small/extended_10x/inference
)

snapshot=(
densenet_small/extended_10x/snapshots/densenet.ckpt-18675
)

mags=( 10 )
sizes=( 256 )
batches=( 25 )
>>>>>>> 2ec4c25d8aee77be3d0b89b54c6e8691dc346b36

for i in `seq 0 ${#outdir[@]}`; do
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model densenet \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size ${batches[$i]} \
  --mag ${mags[$i]} \
  --size ${sizes[$i]} \
  --n_classes 5
done
