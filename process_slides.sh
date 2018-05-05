#!/bin/bash

svslist=$( ls /media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/*svs )
outdir=unet/10x/inference

for svs in ${svslist[@]}; do
  echo $svs $outdir
  python ./deploy_trained.py --slide $svs --out $outdir
done
