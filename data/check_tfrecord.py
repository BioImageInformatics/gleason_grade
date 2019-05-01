from __future__ import print_function

import sys
import argparse
import numpy as np
import tensorflow as tf

sys.path.insert(0, '../tfmodels')
import tfmodels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def check_y(y, n_classes=5):
  """ y is a np array [b, h, w, n_classes] """  
  ymax = np.argmax(y, axis=-1)
  classtotals = {k: 0 for k in np.arange(n_classes)}
  for i in np.arange(ymax.shape[0]):
    y_ = ymax[i,...] 
    for c in np.arange(n_classes):
      classtotals[c] += (y_ == c).sum()
  return classtotals

def check_record(args):
  
  with tf.Session(config=config) as sess:
    dataset = tfmodels.TFRecordImageMask(
      training_record = args.record_path,
      sess = sess,
      crop_size = args.crop_size,
      ratio = args.ratio,
      batch_size = None,
      prefetch = None,
      shuffle_buffer = 64,
      n_classes = args.classes,
      preprocess = [],
      repeat = False, 
      n_threads = args.threads)

    dataset.print_info()

    idx = 0
    classtotals = {k: 0 for k in np.arange(args.classes)}
    while True:
      try:
        x, y = sess.run([dataset.image_op, dataset.mask_op])
        idx += 1
        batchtotals = check_y(y, args.classes)
        for k in np.arange(args.classes):
          classtotals[k] += batchtotals[k]

        if idx == 100:
          break

      except tf.errors.OutOfRangeError:
        print('Reached end of {} examples'.format(idx))
        break

      except:
        break

    total = np.sum([i for _,i in classtotals.items()])
    for k in np.arange(args.classes):
      c = classtotals[k]
      print('Class {}: {} {}'.format(k, c, c/total))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('record_path')
  parser.add_argument('--classes', default=5, type=int)
  parser.add_argument('--crop_size', default=256, type=int)
  parser.add_argument('--ratio', default=1.0, type=float)
  parser.add_argument('--threads', default=4, type=int)

  args = parser.parse_args()

  check_record(args)
