#!/bin/bash

set -e

valjpg=../data/val_jpg
valmask=../data/val_mask

snapshots=(
5x/snapshots/densenet.ckpt-15500
5x/snapshots/densenet.ckpt-31000
5x/snapshots/densenet.ckpt-46500
5x/snapshots/densenet.ckpt-62000
5x/snapshots/densenet.ckpt-77345
10x/snapshots/densenet.ckpt-15500
10x/snapshots/densenet.ckpt-31000
10x/snapshots/densenet.ckpt-46500
10x/snapshots/densenet.ckpt-62000
10x/snapshots/densenet.ckpt-77345
20x/snapshots/densenet.ckpt-15500
20x/snapshots/densenet.ckpt-31000
20x/snapshots/densenet.ckpt-46500
20x/snapshots/densenet.ckpt-62000
20x/snapshots/densenet.ckpt-77345
)

crops=( 1024 1024 1024 1024 1024 512 512 512 512 512 256 256 256 256 256 )
ratios=( 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.5 0.5 1.0 1.0 1.0 1.0 1.0 )

for i in `seq 0 14`; do

  echo $valjpg $valmask ${snapshots[$i]} ${crops[$i]} ${ratios[$i]}
  python test.py $valjpg $valmask ${snapshots[$i]} ${crops[$i]} ${ratios[$i]}

done
