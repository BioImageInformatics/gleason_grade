#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import cv2
import os
import sys
import glob
import argparse

IMGSIZE = 512
N_SUBIMGS = 5
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

  for k in range(5):
    pth = os.path.join(output_base, '{}'.format(k))
    os.makedirs(pth)

CLASSES = range(5)
def main(source_jpg_dir, source_mask_dir, output_base):
  source_jpg_list = sorted(glob.glob(os.path.join(source_jpg_dir, '*jpg')))
  source_mask_list = sorted(glob.glob(os.path.join(source_mask_dir, '*png')))
  class_counts = np.zeros(5)

  if not os.path.exists(output_base):
    initialize_output_dirs(output_base)

  ix = 0
  for img, mask in zip(source_jpg_list, source_mask_list):
    ix += 1
    print(ix, class_counts, img, mask)
    img = cv2.imread(img, -1)
    mask = cv2.imread(mask, -1)
    subimgs = split_subimgs(img)
    submasks = split_subimgs(mask)
    for subimg, submask in zip(subimgs, submasks):
      counts = np.zeros(5)
      for k in CLASSES:
        counts[k] = (submask == k).sum()

      stroma_cnt = counts[4]
      non_stroma_max = np.argmax(counts[:4])
      non_stroma_cnt = counts[non_stroma_max]

      if stroma_cnt > 0.95 * counts.sum():
        outpath = os.path.join(output_base,
          '{}'.format(4),
          '{}.{}.jpg'.format(4, class_counts[4]))
        cv2.imwrite(outpath, subimg)
        class_counts[4] += 1
        continue

      if non_stroma_cnt > 0.5 * counts.sum():
        outpath = os.path.join(output_base,
          '{}'.format(non_stroma_max),
          '{}.{}.jpg'.format(non_stroma_max, int(class_counts[non_stroma_max])))
        cv2.imwrite(outpath, subimg)
        class_counts[non_stroma_max] += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_jpg_dir')
  parser.add_argument('--source_mask_dir')
  parser.add_argument('--output_base')

  args = parser.parse_args()

  main(args.source_jpg_dir, args.source_mask_dir, args.output_base)
