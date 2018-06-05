#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/
# svsdir=/media/ing/D/svs/TCGA_PRAD/
svsdir=data/validation_svs

outdir=(
fcn8s_small/5x/inference
fcn8s_small/10x/inference
fcn8s_small/20x/inference
fcn8s_small/5x_FOV/inference
fcn8s_small/10x_FOV/inference
fcn8s_small/20x_FOV/inference
)

snapshot=(
fcn8s_small/5x/snapshots/fcn.ckpt-26865
fcn8s_small/10x/snapshots/fcn.ckpt-25625
fcn8s_small/20x/snapshots/fcn.ckpt-82585
fcn8s_small/5x_FOV/snapshots/fcn.ckpt-40795
fcn8s_small/10x_FOV/snapshots/fcn.ckpt-51250
fcn8s_small/20x_FOV/snapshots/fcn.ckpt-81795
)

mags=( 5 10 20 5 10 20 )

sizes=( 128 256 512 256 256 256 )

batches=( 20 16 6 16 16 16 )

for i in `seq 0 ${#outdir[@]}`; do
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model fcn8s_s \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size ${batches[$i]} \
  --mag ${mags[$i]} \
  --size ${sizes[$i]}
done
