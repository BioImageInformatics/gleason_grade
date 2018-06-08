#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/
# svsdir=/media/ing/D/svs/TCGA_PRAD/
svsdir=data/validation_svs

outdir=(
densenet_small/extended_10x/inference
)

snapshot=(
densenet_small/extended_10x/snapshots/densenet.ckpt-18675
)

# mags=( 5 10 20 5 10 20 )
# sizes=( 128 256 512 256 256 256 )
# batches=( 20 16 6 16 16 16 )

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
  --size ${sizes[$i]}
done
