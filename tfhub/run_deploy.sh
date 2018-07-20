#!/bin/bash

set -e

# svs_dir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/durham
svs_dir=../data/validation_svs

model_names=(
mobilenet_v2_050_224_50pct
)

for model_name in ${model_names[@]}; do
  outdir="inference/$model_name"
  model_path="snapshots/$model_name"
  python ./deploy_retrained.py --model_path $model_path --slide $svs_dir --out $outdir
done
