from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import (jaccard_similarity_score,
                             confusion_matrix,
                             roc_auc_score,
                             accuracy_score,
                             classification_report,
                             f1_score)
import os, sys, glob, shutil, time, argparse, cv2

sys.path.insert(0, '.')
from fcn8s_small import Inference

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

CROP_SIZE = 1024
RESIZE_FACTOR = 0.25

def compare_tile(y_true_vect, y_hat_vect):
    accuracy = accuracy_score(y_true_vect, y_hat_vect)
    return accuracy


def crop_and_resize(img, mask, crop=CROP_SIZE, resize=RESIZE_FACTOR):
    h, w = img.shape[:2]

    c_h, c_w = h/2, w/2
    crop_half = crop/2
    img = img[c_h-crop_half: c_h+crop_half, c_w-crop_half:c_w+crop_half, :]
    mask = mask[c_h-crop_half: c_h+crop_half, c_w-crop_half:c_w+crop_half]

    img = cv2.resize(img, dsize=(0,0), fx=resize, fy=resize)
    mask = cv2.resize(mask, dsize=(0,0), fx=resize, fy=resize,
        interpolation=cv2.INTER_NEAREST)

    return img, mask


def per_class_metrics(y_true_all, y_hat_all):
    unique_classes = np.unique(y_true_all)
    accuracies = []
    f1 = []
    for c in sorted(unique_classes):
        y_true_c = (y_true_all == c).astype(np.uint8)
        y_hat_c = (y_hat_all == c).astype(np.uint8)
        accuracies.append(accuracy_score(y_true_c, y_hat_c))
        f1.append(f1_score(y_true_c, y_hat_c))

    return accuracies + f1


def test_tiles(jpg_dir, mask_dir, snapshot, crop=CROP_SIZE, resize=RESIZE_FACTOR, outfile=None):
    jpg_patt = os.path.join(jpg_dir, '*.jpg')
    jpg_list = sorted(glob.glob(jpg_patt))

    mask_patt = os.path.join(mask_dir, '*.png')
    mask_list = sorted(glob.glob(mask_patt))

    assert len(jpg_list) == len(mask_list)

    ## Check agreement based on filenames
    for jpg, mask in zip(jpg_list, mask_list):
        jpg_base = os.path.basename(jpg).replace('.jpg', '')
        mask_base = os.path.basename(mask).replace('.png', '')
        assert jpg_base == mask_base, '{} mismatch {}'.format(jpg, mask)
    print('Test files passed agreement check (n = {})'.format(len(jpg_list)))

    aggregate_metrics = []
    indices = []
    y_hat_all = np.array([])
    y_true_all = np.array([])

    x_dims = [int(crop*resize), int(crop*resize), 3]

    with tf.Session(config=config) as sess:
        model = Inference(sess=sess, x_dims=x_dims)
        model.restore(snapshot)

        test_time = time.time()
        for idx, (jpg, mask) in enumerate(zip(jpg_list, mask_list)):
            tile_name = os.path.basename(jpg).replace('.jpg', '')
            indices.append(tile_name)
            img = cv2.imread(jpg)[:,:,::-1]
            y_true = cv2.imread(mask, -1)

            img, y_true = crop_and_resize(img, y_true, crop=crop, resize=resize)
            img = img * (2./255) - 1.

            y_hat = model.inference(np.expand_dims(img, 0))
            y_hat = np.argmax(y_hat, axis=-1)

            y_hat_vect = y_hat.flatten()
            y_true_vect = y_true.flatten()
            y_hat_all = np.concatenate([y_hat_all, y_hat_vect], axis=0)
            y_true_all = np.concatenate([y_true_all, y_true_vect], axis=0)

            aggregate_metrics.append(compare_tile(y_true_vect, y_hat_vect))

    print('\n[\tTest time: {:3.4f}s\t]'.format(time.time() - test_time))

    metrics = per_class_metrics(y_true_all, y_hat_all)
    metric_str = ''
    for metric in metrics:
        metric_str += '{}\t'.format(metric)

    metric_str += '{}\t'.format(accuracy_score(y_true_all, y_hat_all))
    metric_str += '{}\n'.format(f1_score(y_true_all, y_hat_all, average='weighted'))
    output_str = '{}\t'.format(snapshot) + metric_str
    print(output_str)
    outfile.write(output_str)
    outfile.close()


OUTPUT_HEAD = 'SNAPSHOT\tG3_A\tG4_A\tG5_A\tBN_A\tST_A\tG3_F1\tG4_F1\tG5_F1\tBN_F1\tST_F1\tOVERALL_A\tOVERALL_F1\n'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpg_dir')
    parser.add_argument('--mask_dir')
    parser.add_argument('--snapshot')
    parser.add_argument('--mag')
    parser.add_argument('--outfile')
    parser.add_argument('--experiment', default='FOV')
    args = parser.parse_args()

    if args.mag == '5':
        if args.experiment == 'FOV':
            crop = 1024
            resize = 0.25
        elif args.experiment == 'MAG':
            crop = 512
            resize = 0.25

    elif args.mag == '10':
        if args.experiment == 'FOV':
            crop = 512
            resize = 0.5
        elif args.experiment == 'MAG':
            crop = 512
            resize = 0.5

    elif args.mag == '20':
        if args.experiment == 'FOV':
            crop = 256
            resize = 1.
        elif args.experiment == 'MAG':
            crop = 512
            resize = 1.
    else:
        raise Exception('Magnification (--mag) invalid')

    outfile = args.outfile

    if not os.path.exists(outfile):
        outfile = open(outfile, 'w+')
        outfile.write(OUTPUT_HEAD)
    else:
        outfile = open(outfile, 'a')


    test_tiles(args.jpg_dir, args.mask_dir, args.snapshot, crop, resize, outfile)
    outfile.close()
