from __future__ import print_function
import glob, os, shutil, sys, re
import numpy as np


JPG_PATT = './jpg/*.jpg'
MASK_PATT = './mask/*.png'

TRAIN_JPG = './train_jpg'
TRAIN_MASK = './train_mask'
VAL_JPG = './val_jpg'
VAL_MASK = './val_mask'

VAL_PCT = 0.2

def main():
    jpg_list = sorted(glob.glob(JPG_PATT))
    mask_list = sorted(glob.glob(MASK_PATT))
    jpg_mask_zip = zip(jpg_list, mask_list)

    ## Check lists agreement
    for j, m in jpg_mask_zip:
        jb = os.path.basename(j).replace('.jpg', '')
        mb = os.path.basename(m).replace('.png', '')
        if jb != mb:
            print('ERROR: {} / {}'.format(jb, mb))
            return 0
    print('{} pairs passed check'.format(len(j)))

    ## Partition the train/val
    indices = range(len(jpg_mask_zip))
    n_val = int(len(jpg_mask_zip) * VAL_PCT)
    print('Splitting {} validation'.format(n_val))
    print(jpg_mask_zip[0])
    np.random.shuffle(jpg_mask_zip)
    print(jpg_mask_zip[0])

    train_list = jpg_mask_zip[n_val:]
    val_list = jpg_mask_zip[:n_val]

    ## Check output dirs
    for dd in [TRAIN_JPG, TRAIN_MASK, VAL_JPG, VAL_MASK]:
        if not os.path.exists(dd):
            os.makedirs(dd)

    ## Copyfiles
    for j, m in train_list:
        jb = os.path.basename(j)
        mb = os.path.basename(m)
        jnew = os.path.join(TRAIN_JPG, jb)
        mnew = os.path.join(TRAIN_MASK, mb)
        shutil.copyfile(j, jnew)
        shutil.copyfile(m, mnew)

    for j, m in val_list:
        jb = os.path.basename(j)
        mb = os.path.basename(m)
        jnew = os.path.join(VAL_JPG, jb)
        mnew = os.path.join(VAL_MASK, mb)
        shutil.copyfile(j, jnew)
        shutil.copyfile(m, mnew)

    print('Done!')

if __name__ == '__main__':
    main()
