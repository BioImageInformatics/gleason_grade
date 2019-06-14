#!/usr/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import glob
import argparse

from dump_imgs import list_images

def main(args, model_class, sess):

  height = width = 256
  model = model_class(sess=sess, 
                      x_dims=(height, width, 3),
                      n_classes=5)
  model.restore(args.snapshot)

  image_in = model.x_in # A tensorflow op
  bottleneck_op = model.intermediate_ops['05. Bottleneck']
  yhat_op = model.y_hat

  zvect, yvect = [], []
  imglist = list_images(args.source, ext='jpg')
  for idx, imgpath in enumerate(imglist):
    img = cv2.imread(imgpath)[:,:,::-1]
    # img = np.pad(img, ((16, 16), (16, 16), (0, 0)), mode='constant', constant_values=0)
    img = img * (2/255.) - 1.
    img = np.expand_dims(img, 0)

    zhat, yhat = sess.run([bottleneck_op, yhat_op], feed_dict={image_in: img})
    zhat = np.mean(zhat, axis=(1,2))
    # print(yhat.shape, zhat.shape)

    # Retrieve majority class;
    yhat = np.squeeze(np.argmax(yhat, axis=-1))
    if yhat.shape[-1] > args.nPossibleClasses:
      yhat[yhat > 1] -= 1

    if args.savemasks:
      dst = os.path.join(args.dest, 'pred', '{:05d}.png'.format(idx))
      # print('{} --> {}'.format(yhat.shape, dst))
      cv2.imwrite(dst, yhat * (255. / args.nPossibleClasses))

    counts = [np.sum(yhat == c) for c in range(args.nPossibleClasses)]
    perm = np.argsort(counts)
    maj = perm[-1]
    
    if maj == args.nPossibleClasses - 1: # stroma
      ypct = counts[maj] / float((height*width))
      if ypct < 0.9:
        maj = perm[-2]

    yvect.append(maj)
    zvect.append(zhat)

    if idx % 100 == 0:
      print('{} --> {} --> {}'.format(imgpath, img.shape, maj))
      print(counts)

  yvect = np.asarray(yvect)
  ydst = os.path.join(args.dest, 'ypred.npy')
  print('\ny:{}'.format(np.unique(yvect)))
  print('{} --> {}'.format(yvect.shape, ydst))
  np.save(ydst, yvect)

  zvect = np.concatenate(zvect, axis=0)
  zdst = os.path.join(args.dest, 'z.npy')
  print('{} --> {}'.format(zvect.shape, zdst))
  np.save(zdst, zvect)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('source')
  parser.add_argument('dest')
  parser.add_argument('snapshot')
  parser.add_argument('--nPossibleClasses', default=4, type=int)
  parser.add_argument('--collapse_classes', default=True, action='store_false')
  parser.add_argument('--savemasks', default=False, action='store_true')
  args = parser.parse_args()

  if args.savemasks:
    dst = os.path.join(args.dest, 'pred')
    if not os.path.exists(dst):
      os.makedirs(dst)

  sys.path.insert(0, '../../densenet_small')
  from densenet_small import Inference

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    main(args, Inference, sess)