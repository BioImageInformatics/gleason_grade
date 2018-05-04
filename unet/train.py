from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys, datetime, os, time

sys.path.insert(0, '../tfmodels')
import tfmodels

sys.path.insert(0, '.')
from unet import Training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_record_path = '../data/gleason_grade_train.tfrecord'
test_record_path =  '../data/gleason_grade_val.tfrecord'

def main(batch_size, image_ratio, crop_size, n_epochs, lr_0, basedir):
    n_classes = 5
    # batch_size = 32
    # crop_size = 512
    # image_ratio = 0.25
    x_dims = [int(crop_size*image_ratio),
              int(crop_size*image_ratio),
              3]

    iterations = (500/batch_size)*5  ## Define epoch as 10 passes over the data
    epochs = n_epochs ## if epochs=500, then we get 500 * 10 = 2500 times over the data
    snapshot_epochs = 10
    test_epochs = 10
    step_start = 0

    prefetch = 756
    threads = 8

    # basedir = '5x'
    log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(
        basedir=basedir, remove_old=False)
    snapshot_path = None

    gamma = 1e-5
    # lr_0 = 1e-5
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

        if snapshot_path is not None:
            model.restore(snapshot_path)

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
                    # model.train_step(lr=learning_rate(lr_0, gamma, training_step))
                    model.train_step(lr=epoch_lr)
                    # model.train_step(lr=1e-4)

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
    batch_size = int(sys.argv[1])
    image_ratio = float(sys.argv[2])
    crop_size = int(sys.argv[3])
    n_epochs = int(sys.argv[4])
    lr_0 = float(sys.argv[5])
    basedir = sys.argv[6]

    main(batch_size, image_ratio, crop_size, n_epochs, lr_0, basedir)
