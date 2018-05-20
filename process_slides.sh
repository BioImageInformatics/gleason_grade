#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/
svsdir=/media/ing/D/svs/TCGA_PRAD/

outdir=(
fcn8s_small/5x_FOV/inference
fcn8s_small/10x_FOV/inference
fcn8s_small/20x_FOV/inference
unet_small/5x_FOV/inference
unet_small/10x_FOV/inference
)

snapshot=(
fcn8s_small/5x_FOV/snapshots/fcn.ckpt-30845
fcn8s_small/10x_FOV/snapshots/fcn.ckpt-61845
fcn8s_small/20x_FOV/snapshots/fcn.ckpt-116095
unet_small/5x_FOV/snapshots/unet.ckpt-24875
unet_small/10x_FOV/snapshots/unet.ckpt-49875
)

models=(
fcn8s_s
fcn8s_s
fcn8s_s
unet_s
unet_s
)

mags=(
5
10
20
5
10
)

for i in `seq 0 ${#outdir[@]}`; do
  echo ${outdir[$i]} ${snapshot[$i]}
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model ${models[$i]} \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size 12 \
  --mag ${mags[$i]} \
  --size 256
done
