from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import datetime
import os
import time
import argparse

import tfmodels
from gleason_grade import get_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def write_arguments(args):
  argfile = os.path.join(args.basedir, 'params.txt')
  with open(argfile, 'w+') as f:
    for k, i  in sorted(args.__dict__.items()):
      s = '{}:\t{}\n'.format(k, i)
      f.write(s)


def main(args):
  x_dims = [int(args.crop_size * args.image_ratio),
            int(args.crop_size * args.image_ratio),
            3]

  snapshot_epochs = 5
  test_epochs = 25
  step_start = 0

  prefetch = 512
  threads = 8

  log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(
    basedir=args.basedir, remove_old=True)
  write_arguments(args)

  gamma = 1e-5
  def learning_rate(lr_0, gamma, step):
    return lr_0 * np.exp(-gamma*step)

  with tf.Session(config=config) as sess:
    dataset = tfmodels.TFRecordImageMask(
      training_record = args.train_record,
      testing_record = args.val_record,
      sess = sess,
      crop_size = args.crop_size,
      ratio = args.image_ratio,
      batch_size = args.batch_size,
      prefetch = prefetch,
      shuffle_buffer = prefetch,
      n_classes = args.n_classes,
      as_onehot = True,
      mask_dtype = tf.uint8,
      img_channels = 3,
      # preprocess = [],
      n_threads = threads)
    dataset.print_info()

    model_class = get_model(args.model_type, sess, None, None, training=True)
    model = model_class( sess = sess,
      dataset = dataset,
      global_step = step_start,
      learning_rate = args.lr,
      log_dir = log_dir,
      save_dir = save_dir,
      summary_iters = 200,
      summary_image_iters = args.iterations,
      summary_image_n = 4,
      max_to_keep = 25,
      n_classes = args.n_classes,
      # summarize_grads = True,
      # summarize_vars = True,
      x_dims = x_dims)
    model.print_info()

    if args.restore_path is not None:
      model.restore(args.restore_path)

    ## --------------------- Optimizing Loop -------------------- ##
    print('Start')

    try:
      ## Re-initialize training step to have a clean learning rate curve
      training_step = 0
      print('Starting with model at step {}'.format(model.global_step))
      for epx in range(1, args.epochs):
        epoch_start = time.time()
        epoch_lr = learning_rate(args.lr, gamma, training_step)
        for itx in range(args.iterations):
          training_step += 1
          model.train_step(lr=epoch_lr)

        print('Epoch [{}] step [{}] time elapsed [{}]s'.format(
          epx, model.global_step, time.time()-epoch_start))

        # if epx % test_epochs == 0:
        #   model.test(keep_prob=1.0)

        if epx % snapshot_epochs == 0:
          model.snapshot()

    except Exception as e:
      print('Caught exception')
      print(e.__doc__)
      print(e.message)
    finally:
      model.snapshot()
      print('Stopping threads')
      print('Done')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Required - expected to change every time
  parser.add_argument( 'model_type' , type=str)
  parser.add_argument( 'train_record' , type=str)
  parser.add_argument( 'val_record' , type=str)
  parser.add_argument( 'basedir', type=str)

  # TODO replace with the "mag" / process-size parameters
  parser.add_argument( '--image_ratio', default=0.5 , type=float)
  parser.add_argument( '--crop_size', default=512 , type=int)

  # Model stuff
  parser.add_argument( '--n_classes', default=4 , type=int)

  # optimizer settings
  parser.add_argument( '--lr', default=1e-4 , type=float)
  parser.add_argument( '--epochs', default=50 , type=int)
  parser.add_argument( '--iterations', default=1000 , type=int)
  parser.add_argument( '--batch_size' , default=16 , type=int)
  parser.add_argument( '--restore_path', default=None, type=str)

  args = parser.parse_args()

  main(args)
