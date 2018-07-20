#!/bin/bash

set -e

modules=(
# inception_v3
mobilenet_v2_050_224
# nasnet_large
# resnet_v2_152
)

pct=2pct

for module_name in ${modules[@]}; do
  module_url="https://tfhub.dev/google/imagenet/$module_name/feature_vector/1"

  python retrain.py --image_dir ../data/tfhub_ext_${pct} \
  --summaries_dir ./logs/${module_name}_${pct} \
  --bottleneck_dir ./bottlenecks/${module_name}_${pct} \
  --tfhub_module $module_url \
  --saved_model_dir ./snapshots/${module_name}_${pct} \
  --how_many_training_steps 1000 \
  --learning_rate 0.001 \
# --flip_left_right 1
done
