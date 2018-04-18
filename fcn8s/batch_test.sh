#!/bin/bash

set -e

valjpg=../data/val_jpg
valmask=../data/val_mask

snapshots=(
5x/snapshots/fcn.ckpt-74400
10x/snapshots/fcn.ckpt-77345
20x/snapshots/fcn.ckpt-77345
)

crops=( 1024 512 256 )
ratios=( 0.25 0.5 1.0 )

for i in `seq 0 2`; do

  echo $valjpg $valmask ${snapshots[$i]} ${crops[$i]} ${ratios[$i]}
  python test.py $valjpg $valmask ${snapshots[$i]} ${crops[$i]} ${ratios[$i]}

done
