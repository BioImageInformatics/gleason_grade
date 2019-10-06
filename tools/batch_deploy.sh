#!/bin/bash

set -e 

#python Deploy.py docker-slides.txt densenet seg-05x-densenet/snapshots/densenet.ckpt-49000 --mag 5 --chunk 128 --ovr 1.1 -b 32 --suffix .seg.05x.densenet.npy
#python Deploy.py docker-slides.txt densenet seg-10x-densenet/snapshots/densenet.ckpt-49000 --mag 10 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.10x.densenet.npy
#python Deploy.py docker-slides.txt densenet seg-20x-densenet/snapshots/densenet.ckpt-49000 --mag 20 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.20x.densenet.npy
#
#python Deploy.py docker-slides.txt densenet_s seg-05x-densenet-s/snapshots/densenet.ckpt-49000 --mag 5 --chunk 128 --ovr 1.1 -b 32 --suffix .seg.05x.densenet_s.npy
#python Deploy.py docker-slides.txt densenet_s seg-10x-densenet-s/snapshots/densenet.ckpt-49000 --mag 10 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.10x.densenet_s.npy
#python Deploy.py docker-slides.txt densenet_s seg-20x-densenet-s/snapshots/densenet.ckpt-49000 --mag 20 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.20x.densenet_s.npy

#python Deploy.py docker-slides.txt fcn8s seg-05x-fcn8s/snapshots/fcn.ckpt-49000 --mag 5 --chunk 128 --ovr 1.1 -b 32 --suffix .seg.05x.fcn.npy
#python Deploy.py docker-slides.txt fcn8s seg-10x-fcn8s/snapshots/fcn.ckpt-49000 --mag 10 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.10x.fcn.npy
#python Deploy.py docker-slides.txt fcn8s seg-20x-fcn8s/snapshots/fcn.ckpt-49000 --mag 20 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.20x.fcn.npy
#
#python Deploy.py docker-slides.txt fcn8s_s seg-05x-fcn8s-s/snapshots/fcn.ckpt-49000 --mag 5 --chunk 128 --ovr 1.1 -b 32 --suffix .seg.05x.fcn_s.npy
#python Deploy.py docker-slides.txt fcn8s_s seg-10x-fcn8s-s/snapshots/fcn.ckpt-49000 --mag 10 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.10x.fcn_s.npy
#python Deploy.py docker-slides.txt fcn8s_s seg-20x-fcn8s-s/snapshots/fcn.ckpt-49000 --mag 20 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.20x.fcn_s.npy
#
#python Deploy.py docker-slides.txt unet seg-05x-unet/snapshots/unet.ckpt-49000 --mag 5 --chunk 128 --ovr 1.1 -b 16 --suffix .seg.05x.unet.npy
#python Deploy.py docker-slides.txt unet seg-10x-unet/snapshots/unet.ckpt-49000 --mag 10 --chunk 256 --ovr 1.1 -b 16 --suffix .seg.10x.unet.npy
#python Deploy.py docker-slides.txt unet seg-20x-unet/snapshots/unet.ckpt-49000 --mag 20 --chunk 256 --ovr 1.1 -b 16 --suffix .seg.20x.unet.npy

python Deploy.py docker-slides.txt unet_s seg.05x.unet_s/snapshots/unet.ckpt-49000 --mag 5 --chunk 128 --ovr 1.1 -b 32 --suffix .seg.05x.unet_s.npy
python Deploy.py docker-slides.txt unet_s seg.10x.unet_s/snapshots/unet.ckpt-49000 --mag 10 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.10x.unet_s.npy
python Deploy.py docker-slides.txt unet_s seg.20x.unet_s/snapshots/unet.ckpt-49000 --mag 20 --chunk 256 --ovr 1.1 -b 32 --suffix .seg.20x.unet_s.npy

#   # python Example_classifier.py
#   
#   p.add_argument('slides') 
#   p.add_argument('model') # [densenet/_s, fcn8s/_s, unet/_s]
#   p.add_argument('snapshot') 
#   p.add_argument('--iter_type', default='py', type=str) 
#   p.add_argument('--suffix', default='.prob.npy', type=str) 
# 
#   # common arguments with defaults
#   p.add_argument('-b', dest='batchsize', default=64, type=int)
#   p.add_argument('-r', dest='ramdisk', default='/dev/shm', type=str)
#   p.add_argument('-j', dest='workers', default=6, type=int)
#   p.add_argument('-c', dest='n_classes', default=4, type=int)
# 
#   # Slide options
#   p.add_argument('--mag',   dest='process_mag', default=5, type=int)
#   p.add_argument('--chunk', dest='process_size', default=96, type=int)
#   p.add_argument('--bg',    dest='background_speed', default='all', type=str)
#   p.add_argument('--ovr',   dest='oversample_factor', default=1.25, type=float)
#   p.add_argument('--verbose', dest='verbose', default=False, action='store_true')
# 
#   args = p.parse_args()
# 
#   # Functionals for later:
#   args.__dict__['preprocess_fn'] = lambda x: (x * (2/255.)-1).astype(np.float32)
# 
#   tfconfig = tf.ConfigProto()
#   tfconfig.gpu_options.allow_growth = True
#   with tf.Session(config=tfconfig) as sess:
#     main(args, sess)
