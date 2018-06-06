#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/
# svsdir=/media/ing/D/svs/TCGA_PRAD/
svsdir=data/validation_svs

outdir=(
unet/5x/inference
unet/10x/inference
unet/20x/inference
unet/5x_FOV/inference
unet/10x_FOV/inference
unet/20x_FOV/inference
)

snapshot=(
unet/5x/snapshots/unet.ckpt-29450
unet/10x/snapshots/unet.ckpt-61690
unet/20x/snapshots/unet.ckpt-248750
unet/5x_FOV/snapshots/unet.ckpt-58900
unet/10x_FOV/snapshots/unet.ckpt-35875
unet/20x_FOV/snapshots/unet.ckpt-149650
)

mags=( 5 10 20 5 10 20 )

sizes=( 128 256 512 256 256 256 )

batches=( 16 10 2 10 10 10 )

for i in `seq 0 ${#outdir[@]}`; do
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model unet \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size ${batches[$i]} \
  --mag ${mags[$i]} \
  --size ${sizes[$i]}
done
