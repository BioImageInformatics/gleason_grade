#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/
# svsdir=/media/ing/D/svs/TCGA_PRAD/
svsdir=data/validation_svs

outdir=(
densenet/5x/inference
densenet/10x/inference
densenet/20x/inference
densenet/5x_FOV/inference
densenet/10x_FOV/inference
densenet/20x_FOV/inference
)

snapshot=(
densenet/5x/snapshots/densenet.ckpt-30845
densenet/10x/snapshots/densenet.ckpt-55800
densenet/20x/snapshots/densenet.ckpt-187500
densenet/5x_FOV/snapshots/densenet.ckpt-27900
densenet/10x_FOV/snapshots/densenet.ckpt-35650
densenet/20x_FOV/snapshots/densenet.ckpt-102300
)

mags=( 5 10 20 5 10 20 )

sizes=( 128 256 512 256 256 256 )

batches=( 16 12 4 12 12 12 )

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
