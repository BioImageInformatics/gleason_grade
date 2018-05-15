#!/bin/bash

set -e

jpg=../data/val_jpg
mask=../data/val_mask

snapshot_dirs=( 20x/snapshots )

mags=( 20 )

output=test_log_20x.tsv

for i in `seq 0 3`; do
  snapshots=$( ls ${snapshot_dirs[$i]}/*index )
  mag=${mags[$i]}
  for snap in ${snapshots[@]}; do

    echo $snap
    snap=${snap/.index/}
    echo $jpg $mask $snap $mag $output
    python test.py $jpg $mask $snap $mag $output

  done
done
