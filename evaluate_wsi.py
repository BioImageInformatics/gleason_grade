from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
import glob
import cv2
import sys
import os

from sklearn.metrics import (jaccard_similarity_score,
                             confusion_matrix,
                             roc_auc_score,
                             accuracy_score,
                             classification_report,
                             f1_score)

LABEL_CODE = {0: 'G3', 1: 'G4', 2: 'G5', 3: 'BN', 4: 'ST'}

def load_label(label):
    label = cv2.imread(label, -1)
    return label

def load_prediction(pred, argmax=True):
    pred = np.load(pred)
    pred = np.argmax(pred, axis=-1)
    return pred

""" load lists
List out the matching file paths.

Args:
prediciton_dir: base directory for predictions
label_dir: base directory for labels
prediction_patt: file pattern to glob for inside prediction_dir
label_patt: file pattern to glob for inside label_dir

Returns:
prediction_list:
label_list
"""
def load_lists(prediction_dir, label_dir, prediction_patt, label_patt):
    prediction_list = sorted(glob.glob(os.path.join(
        prediction_dir, prediction_patt )))

    pred_base = [os.path.basename(x).replace('_prob.npy', '') for x in prediction_list]
    label_list = [os.path.join(label_dir, '{}.png'.format(x)) for x in pred_base]

    for p, l in zip(prediction_list, label_list):
        if not os.path.exists(p):
            raise Exception('{} does not exist.'.format(p))
        if not os.path.exists(l):
            raise Exception('{} does not exist.'.format(p))
        else:
            print(p ,l)
    return prediction_list, label_list

def resize_label_2_pred(label, pred):
    x_, y_ = pred.shape[:2]
    label = cv2.resize(label, dsize=(y_, x_), interpolation=cv2.INTER_NEAREST)
    return label

DISCARD_STROMA = True
def main(prediction_list, label_list, outfile='wsi_result.csv'):
    colnames = ['Slide', 'Label', 'LabelSize',
        'Pred0', 'Pred1', 'Pred2', 'Pred3', 'PredST',
        'TruePos', 'Majority', 'MajorityPct']
    stats_out = {k: [] for k in colnames}

    for pred_path, label_path in zip(prediction_list, label_list):
        print('\n', pred_path, label_path)
        slide_name = os.path.basename(label_path).replace('.npy', '')
        pred = load_prediction(pred_path)
        label = resize_label_2_pred(load_label(label_path), pred)
        print('\tpred: {} label: {}'.format(pred.shape, label.shape))

        # ## Impose unlabelled area on predictions
        # unlabelled = label == 255
        # pred[unlabelled] == 255

        ## Impose predicted-stroma on the labels
        # stroma = pred == 4
        # label[stroma] = 4

        ## Profile the predicted areas
        ## each slide can have more than one label
        present_labels = np.unique(label)
        for L in present_labels:
            print('\tLabel: {}'.format(L))
            if L == 255:
                continue

            stats_out['Slide'].append(slide_name)
            stats_out['Label'].append(L)

            ## pull out predictions overlapping the label
            label_flat = label.flatten()
            label_ = label_flat[label_flat==L]
            pred_flat = pred.flatten()
            pred_ = pred_flat[label_flat==L]

            print('\tLabel:')
            label_size = (label==L).sum().astype(np.float32)
            print('\t> Class {}: {}'.format(LABEL_CODE[L], label_size))
            stats_out['LabelSize'].append(label_size)

            ## Remove stroma from both
            pred_not_stroma = pred_ != 4
            stroma_area = (pred_ == 4).sum()
            stats_out['PredST'].append(stroma_area)
            print('\tPredicted {} stroma'.format(stroma_area))
            label_ = label_[pred_not_stroma]
            pred_ = pred_[pred_not_stroma]

            print('\tPrediction:')
            present_pred = np.unique(pred_)
            pred_count = np.zeros(4)  # n classes

            for k in range(4):
                ss = (pred_==k).sum()
                print('\t> Class {}: {}'.format(LABEL_CODE[k], ss))
                pred_count[k] = ss
                stats_out['Pred{}'.format(k)].append(ss)

            TP = (pred_ == L).sum()
            acc = TP / label_size
            print('\t######## True Positive: {:3.3f}'.format(acc))

            stats_out['TruePos'].append(TP)

            maj_pred = np.argmax(pred_count)
            maj_percent = pred_count[maj_pred] / pred_count.sum().astype(np.float32)
            print('\t######## Majority: {} ({:3.3f}%)\n'.format(LABEL_CODE[maj_pred], maj_percent))

            stats_out['Majority'].append(maj_pred)
            stats_out['MajorityPct'].append(maj_percent)

    output = pd.DataFrame(stats_out, columns = colnames)
    output.to_csv(outfile)

"""
example:

python evaluate_wsi.py --pred densenet/5x/inference --label data/wsi_annotation/

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred')
    parser.add_argument('--label')
    parser.add_argument('--pred_patt', default='*_prob.npy')
    parser.add_argument('--label_patt', default='*.png')

    args = parser.parse_args()
    prediction_dir = args.pred
    label_dir = args.label
    prediction_patt = args.pred_patt
    label_patt = args.label_patt

    prediction_list, label_list = load_lists(prediction_dir, label_dir,
        prediction_patt, label_patt)

    outfile = os.path.join(prediction_dir, 'performance.csv')
    main(prediction_list, label_list, outfile)
