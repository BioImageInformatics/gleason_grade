#!/usr/bin/env python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import shutil
import glob
import cv2
import os

import MulticoreTSNE as mtsne
from generate_from_img_mask import generate_imgs

# MODEL_URL = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
MODEL_URL = 'https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1'
# MODEL_URL = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1'
# RESNET_V2_101 = 'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1'
# RESNET_V2_152 = 'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1'

IMGOUTDIR = 'imgs'
OUTDIR = 'nasnet-large'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

module = hub.Module(MODEL_URL)
height, width = hub.get_expected_image_size(module)
print(height, width)

image_in = tf.placeholder('float', [1, height, width, 3])
z_op = module(image_in)

sess.run(tf.global_variables_initializer())
JPGDIR = '../../data/tiles_train_val/val_jpg_ext/*.jpg'
MASKDIR = '../../data/tiles_train_val/val_mask_ext/*.png'
jpg_list = sorted(glob.glob(JPGDIR))
mask_list = sorted(glob.glob(MASKDIR))
print(len(jpg_list), len(mask_list))

samples = 5
resize = 0.5
# crop_size = int(height * (1/resize))

generator = generate_imgs(jpg_list, mask_list, samples=samples, 
  resize=resize, crop_size=height)

img_classes, z_vectors = [], []
imglist = []
for img_idx, (x_, maj) in enumerate(generator):
    z = sess.run(z_op, feed_dict={image_in: x_})
    img_classes.append(maj)
    z_vectors.append(z)

    imgout = os.path.join(IMGOUTDIR, '{:03d}.{:02d}.jpg'.format(img_idx, maj))
    cv2.imwrite(imgout, np.squeeze(x_)[:,:,::-1] * 255)
    imglist.append(imgout)
  
z_vectors = np.concatenate(z_vectors, axis=0)
img_classes = np.asarray(img_classes)

z_path = os.path.join(OUTDIR, 'z.npy')
print('z_path:', z_path)
np.save(z_path, z_vectors)

y_path = os.path.join(OUTDIR, 'y.npy')
print('y_path:', y_path)
np.save(y_path, img_classes)

print('images:', len(imglist))
with open('imglist.txt', 'w+') as f:
  for l in imglist:
    f.write('{}\n'.format(l))