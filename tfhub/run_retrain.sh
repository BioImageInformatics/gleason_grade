#!/bin/bash

set -e
# module_name=inception_v3
# module_name=mobilenet_v2_050_224
# module_name=nasnet_large
# module_name=resnet_v2_152

modules=(
inception_v3
# mobilenet_v2_050_224
# nasnet_large
# resnet_v2_152
)

for module_name in ${modules[@]}; do
  module_url="https://tfhub.dev/google/imagenet/$module_name/feature_vector/1"

  python retrain.py --image_dir ../data/tfhub_train \
  --summaries_dir ./logs/$module_name \
  --bottleneck_dir ./bottlenecks/$module_name \
  --tfhub_module $module_url \
  --saved_model_dir ./snapshots/$module_name \
  --how_many_training_steps 5000 \
  --learning_rate 0.001 \
# --flip_left_right 1
done
