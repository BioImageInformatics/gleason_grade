#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/
# svsdir=/media/ing/D/svs/TCGA_PRAD/
svsdir=data/durham_validation

outdir=(
unet_small/5x/inference
unet_small/10x/inference
unet_small/20x/inference
unet_small/5x_FOV/inference
unet_small/10x_FOV/inference
unet_small/20x_FOV/inference
)

snapshot=(
unet_small/5x/snapshots/unet.ckpt-15500
unet_small/10x/snapshots/unet.ckpt-76875
unet_small/20x/snapshots/unet.ckpt-134875
unet_small/5x_FOV/snapshots/unet.ckpt-30845
unet_small/10x_FOV/snapshots/unet.ckpt-42625
unet_small/20x_FOV/snapshots/unet.ckpt-58125
)

mags=( 5 10 20 5 10 20 )

sizes=( 128 256 512 256 256 256 )

batches=( 16 12 4 12 12 12 )

for i in `seq 0 ${#outdir[@]}`; do
  python ./deploy_trained.py \
  --slide_dir $svsdir \
  --model unet_s \
  --out ${outdir[$i]} \
  --snapshot ${snapshot[$i]} \
  --batch_size ${batches[$i]} \
  --mag ${mags[$i]} \
  --size ${sizes[$i]}
done
