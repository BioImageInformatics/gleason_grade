#!/bin/bash

# set -e

jpg=../data/val_jpg_ext
mask=../data/val_mask_ext

snapshot_dir=ext_4class_10x/snapshots
mag=10
output=test_log_exended_4class.tsv

modeldir=densenet_small


cd $modeldir
snapshots=$( ls ${snapshot_dir}/*index )
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
