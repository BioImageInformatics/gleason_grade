#!/bin/bash

set -e
percentages=(
10pct
25pct
50pct
75pct
)
# cd into the model directory
cd densenet_small

for i in ${percentages[@]}; do
  python ./train.py \
  --batch_size 16 \
  --image_ratio 0.5 \
  --crop_size 512 \
  --n_epochs 100 \
  --lr 0.0001 \
  --basedir ext_${i} \
  --train_path ../data/gleason_grade_train_ext.${i}.tfrecord
  # echo ../data/gleason_grade_train_ext.${i}.tfrecord
  # echo $i
done
