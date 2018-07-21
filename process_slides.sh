#!/bin/bash

set -e

svsdir=data/validation_svs

outdir=(
densenet_small/extended_10x/inference
)

snapshot=(
densenet_small/extended_10x/snapshots/densenet.ckpt-18675
)

mags=( 10 )
sizes=( 256 )
batches=( 25 )

for i in `seq 0 ${#outdir[@]}`; do
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model densenet_s \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size ${batches[$i]} \
  --mag ${mags[$i]} \
  --size ${sizes[$i]} \
  --n_classes 5
done
