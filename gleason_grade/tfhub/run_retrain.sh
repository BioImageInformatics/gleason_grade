#!/bin/bash

set -e

modules=(
mobilenet_v2_050_224
)

for module_name in ${modules[@]}; do
  module_url="https://tfhub.dev/google/imagenet/$module_name/feature_vector/1"
  echo $module_url

  python retrain.py --image_dir ../data/tfhub-10x \
  --summaries_dir ./logs/$module_name \
  --bottleneck_dir ./bottlenecks/$module_name \
  --tfhub_module $module_url \
  --saved_model_dir ./snapshots-4class/$module_name \
  --how_many_training_steps 2500 \
  --learning_rate 0.001 \
  # --flip_left_right 1
done
