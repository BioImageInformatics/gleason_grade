#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/

svslist=$( ls $svsdir/*svs )
outdir=fcn8s_small/10x/inference
snapshot=fcn8s_small/10x/snapshots/fcn.ckpt-19900

python ./deploy_trained.py --slide_dir $svsdir --out $outdir --snapshot $snapshot
