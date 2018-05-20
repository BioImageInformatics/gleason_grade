#!/bin/bash

set -e

jpg=/dev/shm/val_jpg
mask=/dev/shm/val_mask

snapshot_dirs=( 5x_FOV/snapshots 10x_FOV/snapshots )

mags=( 5 10 20 )

output=test_log.tsv

for i in `seq 0 3`; do
  snapshots=$( ls ${snapshot_dirs[$i]}/*index )
  mag=${mags[$i]}
  for snap in ${snapshots[@]}; do

    echo $snap
    snap=${snap/.index/}
    echo $jpg $mask $snap $mag $output
    python test.py \
    --jpg_dir $jpg \
    --mask_dir $mask \
    --snapshot $snap \
    --mag $mag \
    --outfile $output \
    --experiment FOV

  done
done
