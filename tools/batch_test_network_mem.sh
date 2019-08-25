#!/bin/bash
set -e

# If this script finishes, none of the nets encouter memory errors 
# and we can expect the long version to finish, eventually
trainrec=../data/gleason_grade_4class_train.tfrecord
valrec=../data/gleason_grade_4class_val.tfrecord
epochs=1
iters=100

python ./train.py densenet $trainrec $valrec seg-05x-densenet --epoch $epoch --iterations $iters --image_ratio 0.25 --crop_size 512 --lr 0.0001 
python ./train.py densenet $trainrec $valrec seg-10x-densenet --epoch $epoch --iterations $iters --image_ratio 0.5 --crop_size 512 --lr 0.0001 
python ./train.py densenet $trainrec $valrec seg-20x-densenet --epoch $epoch --iterations $iters --image_ratio 1.0 --crop_size 256 --lr 0.0001 

python ./train.py densenet_s $trainrec $valrec seg-05x-densenet-s --epoch $epoch --iterations $iters --image_ratio 0.25 --crop_size 512 --lr 0.0001 
python ./train.py densenet_s $trainrec $valrec seg-10x-densenet-s --epoch $epoch --iterations $iters --image_ratio 0.5 --crop_size 512 --lr 0.0001 
python ./train.py densenet_s $trainrec $valrec seg-20x-densenet-s --epoch $epoch --iterations $iters --image_ratio 1.0 --crop_size 256 --lr 0.0001 

python ./train.py fcn8s $trainrec $valrec seg-05x-fcn8s --epoch $epoch --iterations $iters --image_ratio 0.25 --crop_size 512 --lr 0.0001 
python ./train.py fcn8s $trainrec $valrec seg-10x-fcn8s --epoch $epoch --iterations $iters --image_ratio 0.5 --crop_size 512 --lr 0.0001 
python ./train.py fcn8s $trainrec $valrec seg-20x-fcn8s --epoch $epoch --iterations $iters --image_ratio 1.0 --crop_size 256 --lr 0.0001 

python ./train.py fcn8s_s $trainrec $valrec seg-05x-fcn8s-s --epoch $epoch --iterations $iters --image_ratio 0.25 --crop_size 512 --lr 0.0001 
python ./train.py fcn8s_s $trainrec $valrec seg-10x-fcn8s-s --epoch $epoch --iterations $iters --image_ratio 0.5 --crop_size 512 --lr 0.0001 
python ./train.py fcn8s_s $trainrec $valrec seg-20x-fcn8s-s --epoch $epoch --iterations $iters --image_ratio 1.0 --crop_size 256 --lr 0.0001 

python ./train.py unet $trainrec $valrec seg-05x-unet --epoch $epoch --iterations $iters --image_ratio 0.25 --crop_size 512 --lr 0.0001 
python ./train.py unet $trainrec $valrec seg-10x-unet --epoch $epoch --iterations $iters --image_ratio 0.5 --crop_size 512 --lr 0.0001 
python ./train.py unet $trainrec $valrec seg-20x-unet --epoch $epoch --iterations $iters --image_ratio 1.0 --crop_size 256 --lr 0.0001 

python ./train.py unet_s $trainrec $valrec seg-05x-unet_s --epoch $epoch --iterations $iters --image_ratio 0.25 --crop_size 512 --lr 0.0001 
python ./train.py unet_s $trainrec $valrec seg-10x-unet_s --epoch $epoch --iterations $iters --image_ratio 0.5 --crop_size 512 --lr 0.0001 
python ./train.py unet_s $trainrec $valrec seg-20x-unet_s --epoch $epoch --iterations $iters --image_ratio 1.0 --crop_size 256 --lr 0.0001 