#!/bin/bash

set -e

#svsdir=/media/ing/D/svs/TCGA_PRAD

# svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/durham/
svsdir=/media/nathan/d5fd9c1c-4512-4133-a14c-0eada5531282/slide_data/CEDARS_PRAD/

svslist=$( ls $svsdir/*svs )
outdir=unet_small/10x/inference
snapshot=unet_small/10x/snapshots/unet.ckpt-30845

python ./deploy_trained.py --slide_dir $svsdir --out $outdir --snapshot $snapshot
