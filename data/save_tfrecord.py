#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import glob
import sys
import cv2
import os

import tfmodels

img_list_file = 'img_list.txt'
mask_list_file = 'mask_list.txt'

with open(img_list_file, 'r') as f:
    imgs = np.array([x.strip() for x in f])
with open(mask_list_file, 'r') as f:
    masks = np.array([x.strip() for x in f])

print(len(imgs), len(masks))
perm = np.arange(len(imgs))
np.random.shuffle(perm)
imgs = list(imgs[perm])
masks = list(masks[perm])

for k in range(10):
    print(imgs[k], masks[k])

record_path = 'gleason_grade_4class_train.tfrecord'
N_CLASSES = 4
SUBIMG = 512

tfmodels.image_mask_2_tfrecord(imgs, masks, record_path,
   n_classes=N_CLASSES, subimage_size=SUBIMG)

tfmodels.check_tfrecord(record_path, as_onehot=True,
    mask_dtype = tf.uint8, n_classes=N_CLASSES, prefetch=100,
    crop_size=256)
# # tfmodels.check_tfrecord(test_record_path, as_onehot=True,
# #     mask_dtype = tf.uint8, n_classes=N_CLASSES, prefetch=100)
