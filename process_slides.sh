#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/
# svsdir=/media/ing/D/svs/TCGA_PRAD/
svsdir=data/durham_validation

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
densenet/10x/snapshots/densenet.ckpt-61690
densenet/20x/snapshots/densenet.ckpt-200000
densenet/5x_FOV/snapshots/densenet.ckpt-30845
densenet/10x_FOV/snapshots/densenet.ckpt-61845
densenet/20x_FOV/snapshots/densenet.ckpt-116095
)

mags=( 5 10 20 5 10 20 )

sizes=( 128 256 512 256 256 256 )

batches=( 16 8 2 8 8 8 )

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
