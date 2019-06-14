#!/usr/bin/env python

import numpy as np
import cv2
import glob
import shutil
import os

import argparse

def list_images(source, ext='jpg'):
  srch = os.path.join(source, '*.{}'.format(ext))
  return sorted(glob.glob(srch))

def flow_images(imglist):
  pass

def main(args):
  jpg_list = sorted(glob.glob(os.path.join(args.jpgSource, '*.jpg')))
  mask_list = sorted(glob.glob(os.path.join(args.maskSource, '*.png')))

  assert len(jpg_list) == len(mask_list)
  assert len(jpg_list) > 0
  assert len(mask_list) > 0

  if os.path.exists(args.dest) and args.kill:
    print('Removing {}'.format(args.dest))
    shutil.rmtree(args.dest)
  else:
    print('Warning. {} exists'.format(args.dest))

  if not os.path.exists(args.dest):
    os.makedirs(args.dest)

  img_classes, idx = [], 0
  for img_idx, (jpg, mask) in enumerate(zip(jpg_list, mask_list)):
    y = cv2.imread(mask, -1)
    x = cv2.imread(jpg, -1)[:,:,::-1]
    
    ht, wt = x.shape[:2]

    x0_vect = np.linspace(0, ht-args.crop_size, args.samples, dtype=np.int)
    y0_vect = np.linspace(0, wt-args.crop_size, args.samples, dtype=np.int)
    # coords = zip(x0_vect, y0_vect)
              
    for x0 in x0_vect:
      for y0 in y0_vect:

        ## Grab the majority label
        y_ = y[x0:x0+args.crop_size, y0:y0+args.crop_size]
        totals = np.zeros(args.nPossibleClasses)
        for k in range(args.nPossibleClasses):
            totals[k] = (y_==k).sum()
        pcts = totals / np.sum(totals)

        # Check for the presence of a majority
        maj = np.argmax(totals)   
        if pcts[maj] > 0.5:
            # check for stroma -- two ways to skip stroma
            if maj==4 and pcts[maj] < 0.95:
                continue
        else:
            continue

        # Take care of the label shift
        if args.class4 and maj > 1:
          maj -= 1

        img_classes.append(maj)
        x_ = x[x0:x0+args.crop_size, y0:y0+args.crop_size, :]
        x_ = cv2.resize(x_, dsize=(0,0), fx=args.resize, fy=args.resize)

        dst = os.path.join(args.dest, '{:04d}.{:05d}.{:02}.jpg'.format(img_idx, idx, maj))
        idx += 1
        if idx % 250 == 0:
            print('[{:05d} / {:05d}]: {} --> {}'.format(img_idx, len(jpg_list), x_.shape, dst))
        cv2.imwrite(dst, x_)

  img_classes = np.asarray(img_classes)
  ic_dst = os.path.join(args.dest, 'y.npy')
  print('img classes', img_classes.shape)
  print('got classes:', np.unique(img_classes))
  print('saving classes to:', ic_dst)

  np.save(ic_dst, img_classes)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('jpgSource')
  parser.add_argument('maskSource')
  parser.add_argument('dest')
  parser.add_argument('--samples', default=5, type=int)
  parser.add_argument('--resize', default=0.5, type=float)
  parser.add_argument('--crop_size', default=448, type=int)
  parser.add_argument('--nPossibleClasses', default=5, type=int)
  parser.add_argument('--kill', default=False, action='store_true')
  parser.add_argument('--class4', default=False, action='store_true')

  args = parser.parse_args()
  # if args.class4:
  #   args.__setattr__('nPossibleClasses', 4)
  print(args)
  main(args)