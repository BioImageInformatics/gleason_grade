from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import (jaccard_similarity_score,
                             confusion_matrix,
                             roc_auc_score,
                             accuracy_score)
import os, sys, glob, shutil, time, argparse, cv2

sys.path.insert(0, '.')
from model_bayesian import Inference

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

CROP_SIZE = 1024
RESIZE_FACTOR = 0.25
XDIM = [256, 256, 3]
SNAPSHOT = 'snapshots/bayesian_densenet.ckpt-240000'

def compare_tile(yhat_vect, ytrue_vect):

    accuracy = accuracy_score(yhat_vect, ytrue_vect)

    return accuracy


def crop_and_resize(img, mask):
    h, w = img.shape[:2]

    c_h, c_w = h/2, w/2
    crop_half = CROP_SIZE/2
    img = img[c_h-crop_half: c_h+crop_half, c_w-crop_half:c_w+crop_half, :]
    mask = mask[c_h-crop_half: c_h+crop_half, c_w-crop_half:c_w+crop_half]

    img = cv2.resize(img, dsize=(0,0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    mask = cv2.resize(mask, dsize=(0,0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR,
        interpolation=cv2.INTER_NEAREST)

    return img, mask


def test_tiles(jpg_dir, mask_dir):
    jpg_patt = os.path.join(jpg_dir, '*.jpg')
    jpg_list = sorted(glob.glob(jpg_patt))

    mask_patt = os.path.join(mask_dir, '*.png')
    mask_list = sorted(glob.glob(mask_patt))

    assert len(jpg_list) == len(mask_list)

    ## Check agreement based on filenames
    for jpg, mask in zip(jpg_list, mask_list):
        jpg_base = os.path.basename(jpg).replace('.jpg', '')
        mask_base = os.path.basename(mask).replace('.png', '')
        # print(jpg, mask)
        assert jpg_base == mask_base, '{} mismatch {}'.format(jpg, mask)
    print('Test files passed agreement check (n = {})'.format(len(jpg_list)))

    aggregate_metrics = []
    indices = []
    yhat_all = np.array([])
    ytrue_all = np.array([])
    with tf.Session(config=config) as sess:
        model = Inference(sess=sess, x_dims=XDIM)
        model.restore(SNAPSHOT)

        for idx, (jpg, mask) in enumerate(zip(jpg_list, mask_list)):
            print('{:04d}'.format(idx), jpg, mask)
            tile_name = os.path.basename(jpg).replace('.jpg', '')
            indices.append(tile_name)
            img = cv2.imread(jpg)[:,:,::-1]
            ytrue = cv2.imread(mask, -1)

            img, ytrue = crop_and_resize(img, ytrue)
            img = img * (2./255) - 1.

            yhat = model.bayesian_inference(np.expand_dims(img, 0))
            yhat = np.argmax(yhat, axis=-1)

            yhat_vect = yhat.flatten()
            ytrue_vect = ytrue.flatten()
            yhat_all = np.concatenate([yhat_all, yhat_vect], axis=0)
            ytrue_all = np.concatenate([ytrue_all, ytrue_vect], axis=0)

            aggregate_metrics.append(compare_tile(yhat_vect, ytrue_vect))

    aggregated_metrics = pd.DataFrame(aggregate_metrics, index=indices,
        columns=['Accuracy'])
    print(aggregated_metrics)
    confmat = confusion_matrix(yhat_all, ytrue_all)
    print(confmat)

if __name__ == '__main__':
    jpg_dir = sys.argv[1]
    mask_dir = sys.argv[2]

    test_tiles(jpg_dir, mask_dir)