from __future__ import print_function
import cv2
import numpy as np
import glob, sys, os

code_dict = {
  0: 'G3',
  1: 'G4',
  2: 'G5',
  3: 'BN',
  4: 'ST',
}

n_each_class = [0.]*5

mask_patt = './val_mask/*.png'
mask_list = sorted(glob.glob(mask_patt))

for mask_path in mask_list:
    mask = cv2.imread(mask_path, -1)
    print(mask_path, mask.shape)
    #
    mask_copy = np.copy(mask)
    mask[mask_copy==3] = 4
    mask[mask_copy==4] = 3
    cv2.imwrite(mask_path, mask)

    labels, counts = np.unique(mask.ravel(), return_counts=True)

    for label, count in zip(labels, counts):
        print('\t', code_dict[label], count)
        n_each_class[label] += count

with open('label_frequency.txt', 'w+') as f:
    for idx, count in enumerate(n_each_class):
        class_freq = '{}\t{}\n'.format(code_dict[idx], count/np.sum(n_each_class))
        print(class_freq, end='')
        f.write(class_freq)
