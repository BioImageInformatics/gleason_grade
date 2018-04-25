#!/bin/bash

set -e

valjpg=../data/val_jpg
valmask=../data/val_mask

snapshots=(
5x/snapshots/fcn.ckpt-12400
5x/snapshots/fcn.ckpt-37200
5x/snapshots/fcn.ckpt-52700
10x/snapshots/fcn.ckpt-96100
10x/snapshots/fcn.ckpt-111600
10x/snapshots/fcn.ckpt-139500
20x/snapshots/fcn.ckpt-251100
20x/snapshots/fcn.ckpt-275900
20x/snapshots/fcn.ckpt-306900
)

crops=( 1024 1024 1024 512 512 512 256 256 256 )
ratios=( 0.25 0.25 0.25 0.5 0.5 0.5  1.0 1.0 1.0 )
mags=( 5 5 5 10 10 10 20 20 20 )
outfile=fcn_log.tsv

for i in `seq 0 8`; do

  echo $valjpg $valmask ${snapshots[$i]} ${crops[$i]} ${ratios[$i]} ${mags[$i]}
  python test.py $valjpg $valmask ${snapshots[$i]} ${mags[$i]} $outfile

done
