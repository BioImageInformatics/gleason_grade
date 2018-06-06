import tensorflow as tf
import cv2
import numpy as np
import sys, os

sys.path.insert(0,'../tfmodels')
import tfmodels

train_img_patt  = './train_jpg_ext/*.jpg'
train_mask_patt = './train_mask_ext/*.png'
test_img_patt   = './val_jpg_ext/*.jpg'
test_mask_patt  = './val_mask_ext/*.png'

train_record_path = 'gleason_grade_train_ext.tfrecord'
test_record_path = 'gleason_grade_val_ext.tfrecord'
N_CLASSES = 5
SUBIMG = None

tfmodels.image_mask_2_tfrecord(train_img_patt, train_mask_patt, train_record_path,
   n_classes=N_CLASSES, subimage_size=SUBIMG)
tfmodels.image_mask_2_tfrecord(test_img_patt, test_mask_patt, test_record_path,
    n_classes=N_CLASSES, subimage_size=SUBIMG)

tfmodels.check_tfrecord(train_record_path, as_onehot=True,
    mask_dtype = tf.uint8, n_classes=N_CLASSES, prefetch=100)
tfmodels.check_tfrecord(test_record_path, as_onehot=True,
    mask_dtype = tf.uint8, n_classes=N_CLASSES, prefetch=100)
