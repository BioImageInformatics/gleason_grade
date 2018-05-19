from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import os
import time
import argparse

sys.path.insert(0, '../tfmodels')
import tfmodels

sys.path.insert(0, '.')
from fcn8s_small import Training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_record_path = '../data/gleason_grade_train.tfrecord'
test_record_path =  '../data/gleason_grade_val.tfrecord'

def main(batch_size, image_ratio, crop_size, n_epochs, lr_0, basedir, restore_path=None):
    n_classes = 5
    # batch_size = 32
    # crop_size = 512
    # image_ratio = 0.25
    x_dims = [int(crop_size*image_ratio),
              int(crop_size*image_ratio),
              3]

    iterations = (500/batch_size)*5  ## Define epoch as 10 passes over the data
    epochs = n_epochs ## if epochs=500, then we get 500 * 10 = 2500 times over the data
    snapshot_epochs = 25
    test_epochs = 25
    step_start = 0

    prefetch = 2048
    threads = 8

    # basedir = '5x'
    log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(
        basedir=basedir, remove_old=False)

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
            shuffle_buffer = 128,
            n_classes = n_classes,
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
            summary_iters = 100,
            summary_image_iters = iterations,
            summary_image_n = 4,
            max_to_keep = 20,
            # summarize_grads = True,
            # summarize_vars = True,
            x_dims = x_dims)
        model.print_info()

        if restore_path is not None:
            model.restore(restore_path)

        ## --------------------- Optimizing Loop -------------------- ##
        print('Start')
        print('Set up for {} epochs at {} iterations/epoch'.format(epochs, iterations))

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
