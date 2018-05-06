#!/bin/bash

set -e
# module_url=https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1
module_name=inception_v3
# module_url="https://tfhub.dev/google/imagenet/$module_name/feature_vector/1"
module_url="https://tfhub.dev/google/imagenet/$module_name/feature_vector/1"

python retrain.py --image_dir ../data/tfhub_data \
--summaries_dir ./logs/$module_name \
--bottleneck_dir ./bottlenecks/$module_name \
--tfhub_module $module_url \
--saved_model_dir ./snapshots/$module_name \
--how_many_training_steps 1000 \
--learning_rate 0.01 \
--flip_left_right 1
