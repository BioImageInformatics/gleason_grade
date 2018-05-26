#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/
# svsdir=/media/ing/D/svs/TCGA_PRAD/
svsdir=data/durham_validation

outdir=(
fcn8s/5x/inference
fcn8s/10x/inference
fcn8s/20x/inference
fcn8s/5x_FOV/inference
fcn8s/10x_FOV/inference
fcn8s/20x_FOV/inference
)

snapshot=(
fcn8s/5x/snapshots/fcn.ckpt-10500
fcn8s/10x/snapshots/fcn.ckpt-82585
fcn8s/20x/snapshots/fcn.ckpt-237500
fcn8s/5x_FOV/snapshots/fcn.ckpt-23250
fcn8s/10x_FOV/snapshots/fcn.ckpt-54250
fcn8s/20x_FOV/snapshots/fcn.ckpt-112375
)

mags=( 5 10 20 5 10 20 )

sizes=( 128 256 512 256 256 256 )

batches=( 12 8 2 8 8 8 )

for i in `seq 0 ${#outdir[@]}`; do
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model fcn8s \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size ${batches[$i]} \
  --mag ${mags[$i]} \
  --size ${sizes[$i]}
done
