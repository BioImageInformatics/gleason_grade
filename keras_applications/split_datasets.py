#!/usr/bin/env python
import glob
import cv2
import numpy as np

import os
import argparse


def squash_labels(mask):
  mask[mask == 3] = 2
  mask[mask == 4] = 3
  return mask

def split(image, mask, size, overlap):
  x, y = [], []
  h, w = image.shape[:2]

  tgt_h = int(overlap * h / size)
  tgt_w = int(overlap * w / size)

  xx = np.linspace(0 , w - size, tgt_w , dtype=np.int)
  yy = np.linspace(0 , h - size, tgt_h , dtype=np.int)

  for x_ in xx: 
    for y_ in yy:
      subimg = image[y_:y_+size, x_:x_+size, :]
      submask = mask[y_:y_+size, x_:x_+size]


      x.append(subimg)
      y.append(np.argmax(np.bincount(submask.ravel())))

  x = np.stack(x, 0)
  y = np.stack(y, 0)

  return x, y

def load_dataset(image_home, mask_home, patient_list, 
  size = 512, 
  downsample = 0.5, 
  overlap = 1.5, 
  verbose=False):
  """ Load a dataset from image_home using a list of subdirs

  Classes are pulled from mask_home

  Returns 
  x: (n_imgs, h, w, c) 
  y: (n_imgs, n_classes)
  """

  image_list = np.concatenate([sorted(glob.glob(f'{image_home}/{p}/*')) for p in patient_list])
  mask_list = np.concatenate([sorted(glob.glob(f'{mask_home}/{p}/*')) for p in patient_list])

  if verbose:
    for i, (im, m) in enumerate(zip(image_list, mask_list)):
      print(i, im, m)

  x = []
  y = [] 

  for im, m in zip(image_list, mask_list):
    image = cv2.imread(im)[:,:,::-1]
    mask = cv2.imread(m, -1)
    mask = squash_labels(mask)
    
    image = cv2.resize(image, dsize=(0,0), fx=downsample, fy=downsample)
    mask = cv2.resize(mask, dsize=(0,0), fx=downsample, fy=downsample,
      interpolation=cv2.INTER_NEAREST)

    # assert (image.shape == mask.shape).all()
    split_x , split_y = split(image, mask, int(size * downsample), overlap)

    x.append(split_x)
    y.append(split_y)


  x = np.concatenate(x, axis=0)
  y = np.concatenate(y, axis=0)
  y = np.eye(N=y.shape[0], M=4)[y]

  shuffle = np.arange(x.shape[0]).astype(np.int)
  np.random.shuffle(shuffle)
  x = x[shuffle, :]
  y = y[shuffle, :]

  x = (x / 255.).astype(np.float32)

  print('split_datasets returning x:', x.shape, x.dtype, x.min(), x.max())
  print('split_datasets returning y:', y.shape, y.dtype)
  return x, y


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('image_home')
  parser.add_argument('mask_home')
  parser.add_argument('patient_list')

  args = parser.parse_args()
  image_home = args.image_home
  mask_home = args.mask_home

  patient_list = [p.strip() for p in open(args.patient_list, 'r')]

  x, y = load_dataset(image_home, mask_home, patient_list, size=512, overlap=1.2)

  # print('x', x.shape)
  # print('y', y.shape)
