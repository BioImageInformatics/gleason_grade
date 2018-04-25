#!/bin/bash

set -e

valjpg=../data/val_jpg
valmask=../data/val_mask

snapshots=(
5x/snapshots/unet.ckpt-40300
5x/snapshots/unet.ckpt-58900
5x/snapshots/unet.ckpt-61380
10x/snapshots/unet.ckpt-99200
10x/snapshots/unet.ckpt-102300
10x/snapshots/unet.ckpt-148800
20x/snapshots/unet.ckpt-272800
20x/snapshots/unet.ckpt-297600
20x/snapshots/unet.ckpt-309380
)

crops=( 1024 1024 1024 512 512 512 256 256 256 )
ratios=( 0.25 0.25 0.25 0.5 0.5 0.5  1.0 1.0 1.0 )
mags=( 5 5 5 10 10 10 20 20 20 )
outfile=unet_log.tsv

for i in `seq 0 8`; do

  echo $valjpg $valmask ${snapshots[$i]} ${crops[$i]} ${ratios[$i]} ${mags[$i]}
  python test.py $valjpg $valmask ${snapshots[$i]} ${mags[$i]} $outfile

done
