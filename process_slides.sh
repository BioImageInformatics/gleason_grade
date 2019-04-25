#!/bin/bash

set -e

# svsdir=data/validation_svs
svsdir=/mnt/slowdata/slide_data/durham/max_high_grade_content/

outdir=(
densenet/ext_10x/durham_val
)

snapshot=(
densenet/ext_10x/snapshots/densenet.ckpt-1996000
)

mags=( 10 )
sizes=( 256 )
batches=( 6 )

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
