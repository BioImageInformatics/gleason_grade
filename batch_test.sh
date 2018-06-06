#!/bin/bash

# set -e

jpg=../data/val_jpg
mask=../data/val_mask
# jpg=/media/ing/D/image_data/segmentation/gleason_grade/cbm_split/val_jpg
# mask=/media/ing/D/image_data/segmentation/gleason_grade/cbm_split/val_mask

# snapshot_dirs=( 5x_FOV/snapshots 10x_FOV/snapshots 20x_FOV/snapshots )
# mags=( 5 10 20 )
snapshot_dirs=( 20x/snapshots )
mags=( 20 )
output=test_log_MAG.tsv

modeldirs=(
unet
)

for dd in ${modeldirs[@]}; do

  cd /media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/project_data/gleason_grade/$dd
  # cd /home/ing/projects/gleason_grade/$dd
  pwd
  for i in `seq 0 3`; do
    snapshots=$( ls ${snapshot_dirs[$i]}/*index )
    mag=${mags[$i]}
    for snap in ${snapshots[@]}; do

      snapshot=${snap/.index/} # replaces .index with a blank string

      python test.py \
      --jpg_dir $jpg \
      --mask_dir $mask \
      --snapshot $snapshot \
      --mag $mag \
      --outfile $output \
      --experiment MAG

    done
  done

done
