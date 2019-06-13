import tensorflow as tf
import numpy as np
import glob
import sys
import cv2
import os

import tfmodels

train_img_patt  = './x0002/4class/*_rgb.jpg'
train_mask_patt = './x0002/4class/*_mask.png'
# test_img_patt   = './val_jpg_ext/*.jpg'
# test_mask_patt  = './val_mask_ext/*.png'

imgs  = sorted(glob.glob(train_img_patt))
masks = sorted(glob.glob(train_mask_patt)) 
print(len(imgs), len(masks))

train_record_path = 'gleason_grade_4class_x0002.tfrecord'
# test_record_path = 'gleason_grade_val_ext.tfrecord'
N_CLASSES = 4
SUBIMG = 512

tfmodels.image_mask_2_tfrecord(train_img_patt, train_mask_patt, train_record_path,
   n_classes=N_CLASSES, subimage_size=SUBIMG)
# tfmodels.image_mask_2_tfrecord(test_img_patt, test_mask_patt, test_record_path,
#     n_classes=N_CLASSES, subimage_size=SUBIMG)

tfmodels.check_tfrecord(train_record_path, as_onehot=True,
    mask_dtype = tf.uint8, n_classes=N_CLASSES, prefetch=100,
    crop_size=256)
# tfmodels.check_tfrecord(test_record_path, as_onehot=True,
#     mask_dtype = tf.uint8, n_classes=N_CLASSES, prefetch=100)
