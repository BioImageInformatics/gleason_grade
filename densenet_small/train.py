from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import datetime
import os
import time
import argparse

sys.path.insert(0, '../tfmodels')
import tfmodels

sys.path.insert(0, '.')
from densenet_small import Training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

TRAIN_RECORD_PATH = '../data/gleason_grade_train_ext.10pct.tfrecord'
TEST_RECORD_PATH =  '../data/gleason_grade_val_ext.tfrecord'

def main(batch_size, image_ratio, crop_size, n_epochs, lr_0, basedir, restore_path,
         train_record_path=TRAIN_RECORD_PATH, test_record_path=TEST_RECORD_PATH):
    n_classes = 5
    x_dims = [int(crop_size*image_ratio),
              int(crop_size*image_ratio),
              3]

    iterations = 1000
    epochs = n_epochs
    snapshot_epochs = 10
    test_epochs = 10
    step_start = 0

    prefetch = 2048
    threads = 8

    log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(
        basedir=basedir, remove_old=False)
    snapshot_path = None

    gamma = 1e-5
    def learning_rate(lr_0, gamma, step):
        return lr_0 * np.exp(-gamma*step)

    with tf.Session(config=config) as sess:
        dataset = tfmodels.TFRecordImageMask(
            training_record = train_record_path,
            testing_record = test_record_path,
            sess = sess,
            crop_size = crop_size,
            ratio = image_ratio,
            batch_size = batch_size,
            prefetch = prefetch,
            shuffle_buffer = 256,
            n_classes = n_classes,
            as_onehot = True,
            mask_dtype = tf.uint8,
            img_channels = 3,
            n_threads = threads)
        dataset.print_info()

        model = Training( sess = sess,
            dataset = dataset,
            global_step = step_start,
            learning_rate = lr_0,
            log_dir = log_dir,
            save_dir = save_dir,
            summary_iters = 100,
            summary_image_iters = iterations,
            summary_image_n = 4,
            max_to_keep = 50,
            # summarize_grads = True,
            # summarize_vars = True,
            x_dims = x_dims)
        model.print_info()

        if restore_path is not None:
            model.restore(restore_path)

        ## --------------------- Optimizing Loop -------------------- ##
        print('Start')

        try:
            ## Re-initialize training step to have a clean learning rate curve
            training_step = 0
            print('Starting with model at step {}'.format(model.global_step))
            for epx in xrange(1, epochs):
                epoch_start = time.time()
                epoch_lr = learning_rate(lr_0, gamma, training_step)
                for itx in xrange(iterations):
                    training_step += 1
                    model.train_step(lr=epoch_lr)

                print('Epoch [{}] step [{}] time elapsed [{}]s'.format(
                    epx, model.global_step, time.time()-epoch_start))

                if epx % test_epochs == 0:
                    model.test(keep_prob=1.0)

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
    parser.add_argument( '--batch_size' , default=12 , type=int)
    parser.add_argument( '--image_ratio', default=0.25 , type=float)
    parser.add_argument( '--crop_size', default=512 , type=int)
    parser.add_argument( '--n_epochs', default=200 , type=int)
    parser.add_argument( '--lr', default=1e-4 , type=float)
    parser.add_argument( '--basedir', default='trained' , type=str)
    parser.add_argument( '--restore_path', default=None , type=str)
    parser.add_argument( '--train_path', default=None , type=str)
    parser.add_argument( '--test_path', default=None , type=str)

    args = parser.parse_args()
    batch_size = args.batch_size
    image_ratio = args.image_ratio
    crop_size = args.crop_size
    n_epochs = args.n_epochs
    lr = args.lr
    basedir = args.basedir
    restore_path = args.restore_path

    if args.train_path and args.test_path:
        main(batch_size, image_ratio, crop_size, n_epochs, lr, basedir, restore_path,
             train_record_path=args.train_path,
             test_record_path=args.test_path)
    elif args.train_path and not args.test_path:
        main(batch_size, image_ratio, crop_size, n_epochs, lr, basedir, restore_path,
             train_record_path=args.train_path)
    elif args.test_path and not args.train_path:
        main(batch_size, image_ratio, crop_size, n_epochs, lr, basedir, restore_path,
             test_record_path=args.test_path)
    else:
        main(batch_size, image_ratio, crop_size, n_epochs, lr, basedir, restore_path)
