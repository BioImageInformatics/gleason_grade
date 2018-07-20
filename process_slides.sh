#!/bin/bash

set -e

svsdir=data/validation_svs

outdir=(
densenet_small/ext_75pct/inference
)

snapshot=(
densenet_small/ext_75pct/snapshots/densenet.ckpt-99000
)

mags=( 10 )
sizes=( 256)
batches=( 8 )

for i in `seq 0 ${#outdir[@]}`; do
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model densenet_s \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size ${batches[$i]} \
  --mag ${mags[$i]} \
  --size ${sizes[$i]}
done
