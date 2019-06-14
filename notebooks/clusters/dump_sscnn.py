#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import glob

sys.path.insert(0, '../../densenet')
from densenet import Inference

SNAPSHOT = '../../densenet/ext_10x/snapshots/densenet.ckpt-88357'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


height = width = 256
model = Inference(sess=sess, x_dims=(height, width, 3))
model.restore(SNAPSHOT)

image_in = model.x_in
bottleneck_op = model.intermediate_ops['05. Bottleneck']
yhat_op = model.y_hat

img_plotting = {}
img_classes = []
orig_imgs = []

z_vectors = []
y_vectors = []

resize = 0.5
crop_size = int(height * (1/resize))
print(crop_size)

samples = 5
np.random.seed(999)
x_samples = [np.random.randint(0, 1200-crop_size) for _ in range(samples)]
y_samples = [np.random.randint(0, 1200-crop_size) for _ in range(samples)]


jpg_list = sorted(glob.glob('../../data/tiles_train_val/val_jpg_ext/*.jpg'))
mask_list = sorted(glob.glob('../../data/tiles_train_val/val_mask_ext/*.png'))
print(len(jpg_list), len(mask_list))

idx = 0
for img_idx, (jpg, mask) in enumerate(zip(jpg_list, mask_list)):
  y = cv2.imread(mask, -1)
  x = cv2.imread(jpg, -1)[:,:,::-1]
         
  for k in range(samples):
    x0 = x_samples[k]
    y0 = y_samples[k]

    ## Grab the majority label
    y_ = y[x0:x0+crop_size, y0:y0+crop_size]
    totals = np.zeros(5)
    for k in range(5):
      totals[k] = (y_==k).sum()

    # Check for majority
    maj = np.argmax(totals)   
    if totals[maj] > 0.5 * (crop_size**2):
      # check for stroma -- two ways to skip stroma
      if maj==4 and totals[maj] < 0.95 * (crop_size*2):
        continue
    else:
      continue

    img_classes.append(maj)
    orig_imgs.append(img_idx)
    
    idx += 1
    if idx % 500 == 0:
      print('{} [{} / {}]'.format(idx, img_idx, len(jpg_list)))
    x_ = x[x0:x0+crop_size, y0:y0+crop_size, :]
    x_ = cv2.resize(x_, dsize=(0,0), fx=resize, fy=resize)
    x_ = x_ * (2/255.) - 1
    x_ = np.expand_dims(x_, 0)
    
    if np.random.randn() < -3:
      img_plotting[idx] = (x_ + 1)/2.
    
    z, yhat = sess.run([bottleneck_op, yhat_op], feed_dict={image_in: x_, model.keep_prob: 1.})
    ymax = np.argmax(yhat, axis=-1)
    num_classes = np.zeros(5)
    for ck in range(5):
      num_classes[ck] = (ymax==ck).sum()
    
    y_vectors.append(num_classes)
    
    z_int = np.mean(z, axis=(1,2))
    z_vectors.append(z_int)
    
  
z_vectors = np.concatenate(z_vectors, axis=0)

img_classes = np.asarray(img_classes)
orig_imgs = np.asarray(orig_imgs)
print('z vectors', z_vectors.shape)
print('img classes', img_classes.shape)
print('classes:', np.unique(img_classes))


OUTDIR = 'tsne/densenet'
z_path = os.path.join(OUTDIR, 'z.npy')
print('z_path', z_path)
np.save(z_path, z_vectors)

y_path = os.path.join(OUTDIR, 'y.npy')
print('y_path', y_path)
np.save(y_path, img_classes)

y_path = os.path.join(OUTDIR, 'y.npy')
print('y_path', y_path)
np.save(y_path, img_classes)
