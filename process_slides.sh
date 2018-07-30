#!/bin/bash

set -e

svsdir=data/validation_svs

outdir=(
densenet/ext_5x/inference
)

snapshot=(
densenet/ext_5x/snapshots/densenet.ckpt-355669
)

mags=( 5 )
sizes=( 256 )
batches=( 8 )

for i in `seq 0 ${#outdir[@]}`; do
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model densenet \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size ${batches[$i]} \
  --mag ${mags[$i]} \
  --size ${sizes[$i]}
done
