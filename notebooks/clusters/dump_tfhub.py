#!/usr/bin/env python

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import sys
import os
import argparse

from dump_imgs import list_images

def main(args, sess):

  module = hub.Module(args.url)
  height, width = hub.get_expected_image_size(module)
  print(height, width)

  image_in = tf.placeholder('float', [1, height, width, 3])
  z_op = module(image_in)
  sess.run(tf.global_variables_initializer())  

  zvect = []
  imglist = list_images(args.source, ext='jpg')
  for idx, imgpath in enumerate(imglist):
    img = cv2.imread(imgpath)[:,:,::-1]
    img = np.expand_dims(img / 255., 0)

    z = sess.run(z_op, {image_in: img})
    zvect.append(z) 

    if idx % 100 == 0:
      print('{} --> {} --> {}'.format(imgpath, img.shape, z.shape))

  zvect = np.concatenate(zvect, axis=0)
  dst = os.path.join(args.dest, 'z.npy')
  np.save(dst, zvect)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('source')
  parser.add_argument('dest')
  parser.add_argument('--url', default='https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/1')

  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    main(args, sess)
