#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/

svslist=$( ls $svsdir/*svs )
outdir=densenet_small/20x/inference
snapshot=densenet_small/20x/snapshots/densenet.ckpt-124375

python ./deploy_trained.py --slide_dir $svsdir --out $outdir --snapshot $snapshot
