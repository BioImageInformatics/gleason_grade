#!/bin/bash

set -e

problists=(
  seg.05x.densenet
  seg.10x.densenet
  seg.20x.densenet
  seg.05x.densenet_s
  seg.10x.densenet_s
  seg.20x.densenet_s
  seg.05x.fcn
  seg.10x.fcn
  seg.20x.fcn
  seg.05x.fcn_s
  seg.10x.fcn_s
  seg.20x.fcn_s
  seg.05x.unet
  seg.10x.unet
  seg.20x.unet
  seg.05x.unet_s
  seg.10x.unet_s
  seg.20x.unet_s
)

slides=slides.txt
masks=masks.txt

probbase=/mnt/slowdata/gleason-grade-slides

for p in ${problists[@]}; do
  echo $p
  ls $probbase/*${p}.npy > $p/probs.txt
  
  echo $p/probs.txt
  cat $p/probs.txt | wc -l
  # python Overlay_segmentation.py slides.txt $p/probs.txt

  outpath=${p}/eval.csv
  python Evaluate.py $p/probs.txt ${masks} ${outpath} --verbose

done