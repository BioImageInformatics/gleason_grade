#!/bin/bash

set -e

jpg=../data/train_jpg
mask=../data/train_mask

snapshots=(
5x/snapshots/densenet.ckpt-15500
5x/snapshots/densenet.ckpt-23250
5x/snapshots/densenet.ckpt-30690
10x/snapshots/densenet.ckpt-48050
10x/snapshots/densenet.ckpt-68200
10x/snapshots/densenet.ckpt-77190
20x/snapshots/densenet.ckpt-60450
20x/snapshots/densenet.ckpt-80600
20x/snapshots/densenet.ckpt-88350
)

crops=( 1024 1024 1024 512 512 512 256 256 256 )
ratios=( 0.25 0.25 0.25 0.5 0.5 0.5 1.0 1.0 1.0 )

mags=( 5 5 5 10 10 10 20 20 20 )

output=train_log.tsv

for i in `seq 0 14`; do

  echo $jpg $mask ${snapshots[$i]} ${crops[$i]} ${ratios[$i]} ${mags[$i]}
  python test.py $jpg $mask ${snapshots[$i]} ${mags[$i]} $output

done
