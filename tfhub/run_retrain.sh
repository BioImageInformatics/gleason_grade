#!/bin/bash

set -e

python retrain.py --image_dir ../data/tfhub_data \
--output_labels output_labels.txt \
--summaries_dir ./logs \
--bottleneck_dir ./bottlenecks \
--final_tensor_name y_hat \
--tfhub_modue https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1 \
--saved_model_dir ./snapshots
