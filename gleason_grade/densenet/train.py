from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import datetime
import os
import time
import argparse

import tfmodels

sys.path.insert(0, '.')
from densenet import Training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_record_path = '../data/gleason_grade_4class_x0001.tfrecord'
#test_record_path =  '../data/gleason_grade_val_ext.tfrecord'
N_CLASSES = 4

def main(batch_size, image_ratio, crop_size, n_epochs, lr_0, basedir, restore_path):
    n_classes = N_CLASSES
    x_dims = [int(crop_size*image_ratio),
              int(crop_size*image_ratio),
              3]

    iterations = 5000  ## Define epoch length
    epochs = n_epochs ## if epochs=500, then we get 500 * 10 = 2500 times over the data
    snapshot_epochs = 5
    test_epochs = 25
    step_start = 0

    prefetch = 512
    threads = 8

    # basedir = '5x'
    log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(
        basedir=basedir, remove_old=True)

    gamma = 1e-5
    # lr_0 = 1e-5
    def learning_rate(lr_0, gamma, step):
        return lr_0 * np.exp(-gamma*step)

    with tf.Session(config=config) as sess:
        dataset = tfmodels.TFRecordImageMask(
            training_record = train_record_path,
            # testing_record = test_record_path,
            sess = sess,
            crop_size = crop_size,
            ratio = image_ratio,
            batch_size = batch_size,
            prefetch = prefetch,
            shuffle_buffer = 256,
            n_classes = N_CLASSES,
            as_onehot = True,
            mask_dtype = tf.uint8,
            img_channels = 3,
            # preprocess = [],
            n_threads = threads)
        dataset.print_info()

        model = Training( sess = sess,
            dataset = dataset,
            global_step = step_start,
            learning_rate = lr_0,
            log_dir = log_dir,
            save_dir = save_dir,
            summary_iters = 200,
            summary_image_iters = iterations,
            summary_image_n = 4,
            max_to_keep = 25,
            n_classes = N_CLASSES,
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
            for epx in range(1, epochs):
                epoch_start = time.time()
                epoch_lr = learning_rate(lr_0, gamma, training_step)
                for itx in range(iterations):
                    training_step += 1
                    # model.train_step(lr=learning_rate(lr_0, gamma, training_step))
                    model.train_step(lr=epoch_lr)
                    # model.train_step(lr=1e-4)

                print('Epoch [{}] step [{}] time elapsed [{}]s'.format(
                    epx, model.global_step, time.time()-epoch_start))

                # if epx % test_epochs == 0:
                #     model.test(keep_prob=1.0)

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
    parser.add_argument( '--batch_size' , default=16 , type=int)
    parser.add_argument( '--image_ratio', default=1. , type=float)
    parser.add_argument( '--crop_size', default=256 , type=int)
    parser.add_argument( '--n_epochs', default=25 , type=int)
    parser.add_argument( '--lr', default=1e-6 , type=float)
    parser.add_argument( '--basedir', default='5x_x0002' , type=str)
    parser.add_argument( '--restore_path', default='5x_LONG_x0001/snapshots/densenet.ckpt-45000', 
                         type=str)

    # restore_path = '10x/snapshots/unet.ckpt-61690'
    args = parser.parse_args()
    batch_size = args.batch_size
    image_ratio = args.image_ratio
    crop_size = args.crop_size
    n_epochs = args.n_epochs
    lr = args.lr
    basedir = args.basedir
    restore_path = args.restore_path

    main(batch_size, image_ratio, crop_size, n_epochs, lr, basedir, restore_path)
