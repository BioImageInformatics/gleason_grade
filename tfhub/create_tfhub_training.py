from __future__ import print_function
import numpy as np
import cv2
import os
import sys
import glob
import argparse


IMGSIZE = 512
N_SUBIMGS = 5
N_CLASSES = 5
CLASSES = range(N_CLASSES)
def split_subimgs(img):
    x, y = img.shape[:2]
    x_vect = np.linspace(0, x-IMGSIZE, N_SUBIMGS, dtype=np.int)
    y_vect = np.linspace(0, y-IMGSIZE, N_SUBIMGS, dtype=np.int)

    subimgs = []
    for x_ in x_vect:
        for y_ in y_vect:
            try:
                subimgs.append(img[x_:x_+IMGSIZE, y_:y_+IMGSIZE, :])
            except:
                subimgs.append(img[x_:x_+IMGSIZE, y_:y_+IMGSIZE])

    return subimgs

def initialize_output_dirs(output_base):
    os.makedirs(output_base)

    for k in CLASSES:
        pth = os.path.join(output_base, '{}'.format(k))
        os.makedirs(pth)

def renumberMask(mask):
    mask2 = np.copy(mask)
    mask2[mask == 2] = 1
    return mask2


def main(source_jpg_dir, source_mask_dir, output_base, sample_pct=None):
    source_jpg_list = sorted(glob.glob(os.path.join(source_jpg_dir, '*jpg')))
    source_mask_list = sorted(glob.glob(os.path.join(source_mask_dir, '*png')))
    class_counts = np.zeros(N_CLASSES)

    if not os.path.exists(output_base):
        initialize_output_dirs(output_base)

    tmp_join = zip(source_jpg_list, source_mask_list)
    list_len = len(tmp_join)
    print('Images: {}'.format(list_len))
    if sample_pct:
        sample_n = int(list_len * float(sample_pct))
        print('Sampling image list: {} --> {}'.format(list_len, sample_n))
        np.random.shuffle(tmp_join)
        tmp_join = tmp_join[:sample_n]
        print('Verify new length = {}'.format(len(tmp_join)))

    ix = 0
    for img, mask in tmp_join:
        ix += 1
        print(ix, class_counts, img, mask)
        img = cv2.imread(img, -1)
        mask = cv2.imread(mask, -1)

        # Replace 2 with 1
        mask = renumberMask(mask)

        subimgs = split_subimgs(img)
        submasks = split_subimgs(mask)
        for subimg, submask in zip(subimgs, submasks):
            counts = np.zeros(N_CLASSES)
            for k in CLASSES:
                counts[k] = (submask == k).sum()

            stroma_cnt = counts[N_CLASSES-1]
            non_stroma_max = np.argmax(counts[:-1])
            non_stroma_cnt = counts[non_stroma_max]

            if stroma_cnt > 0.95 * counts.sum():
                outpath = os.path.join(output_base,
                    '{}'.format(N_CLASSES-1),
                    '{}.jpg'.format(class_counts[N_CLASSES-1]))
                cv2.imwrite(outpath, subimg)
                class_counts[N_CLASSES-1] += 1
                continue

            if non_stroma_cnt > 0.5 * counts.sum():
                outpath = os.path.join(output_base,
                    '{}'.format(non_stroma_max),
                    '{}.jpg'.format(class_counts[non_stroma_max]))
                cv2.imwrite(outpath, subimg)
                class_counts[non_stroma_max] += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_jpg_dir', default='../data/train_jpg_ext')
    parser.add_argument('--source_mask_dir', default='../data/train_mask_ext')
    parser.add_argument('--output_base', default='../data/tfhub_data')
    parser.add_argument('--sample_pct', type=float, default=1.0)

    args = parser.parse_args()
    source_jpg_dir = args.source_jpg_dir
    source_mask_dir = args.source_mask_dir
    output_base = args.output_base

    main(source_jpg_dir, source_mask_dir, output_base, sample_pct=args.sample_pct)
