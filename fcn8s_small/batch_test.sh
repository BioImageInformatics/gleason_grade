#!/bin/bash

set -e

# jpg=../data/val_jpg
# mask=../data/val_mask
jpg=/media/ing/D/image_data/segmentation/gleason_grade/cbm_split/val_jpg
mask=/media/ing/D/image_data/segmentation/gleason_grade/cbm_split/val_mask

snapshot_dirs=( 5x_FOV/snapshots 10x_FOV/snapshots 20x_FOV/snapshots )

mags=( 5 10 20 )

output=test_log_FOV.tsv

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
