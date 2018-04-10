import numpy as np
import tensorflow as tf
import sys, datetime, os, time

sys.path.insert(0, './tfmodels')
import tfmodels

sys.path.insert(0, '.')
from model_bayesian import Training
from dataset import TFRecordInput

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_record_path = 'ccrcc_big_train.tfrecord'
test_record_path =  'ccrcc_big_test.tfrecord'
batch_size = 16
crop_size = 512
image_ratio = 0.25

epochs = 500
iterations = 3000
snapshot_epochs = 10
step_start = 0

prefetch = 512
threads = 4

basedir = 'b_ccrcc_20180406'
log_dir, save_dir, debug_dir, infer_dir = tfmodels.make_experiment(
    basedir=basedir, remove_old=False)
snapshot_path = None

gamma = 1e-5
lr_0 = 1e-4
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
        n_classes = 4,
        as_onehot = True,
        mask_dtype = tf.uint8,
        img_channels = 3,
        # preprocess = [],
        n_threads = threads)
    dataset.print_info()

    model = Training( sess = sess,
        class_weights = [0.2, 0.2, 0.5, 0.1],
        dataset = dataset,
        dense_stacks = [8, 8, 8, 8],
        growth_rate = 64,
        k_size = 3,
        global_step = step_start,
        learning_rate = lr_0,
        n_classes = 4,
        log_dir = log_dir,
        save_dir = save_dir,
        summary_iters = 100,
        summary_image_iters = 500,
        summary_image_n = 2,
        # summarize_grads = True,
        # summarize_vars = True,
        x_dims = [128, 128, 3])
    model.print_info()

    if snapshot_path is not None:
        model.restore(snapshot_path)

    ## --------------------- Optimizing Loop -------------------- ##
    print 'Start'

    try:
        ## Re-initialize training step to have a clean learning rate curve
        training_step = 0
        print 'Starting with model at step {}'.format(model.global_step)
        for epx in xrange(1, epochs):
            epoch_start = time.time()
            epoch_lr = learning_rate(lr_0, gamma, training_step)
            for itx in xrange(iterations):
                training_step += 1
                # model.train_step(lr=learning_rate(lr_0, gamma, training_step))
                model.train_step(lr=epoch_lr)
                # model.train_step(lr=1e-4)

            model.test(keep_prob=0.8)
            print 'Epoch [{}] step [{}] time elapsed [{}]s'.format(
                epx, model.global_step, time.time()-epoch_start)

            if epx % snapshot_epochs == 0:
                model.snapshot()

    except Exception as e:
        print 'Caught exception'
        print e.__doc__
        print e.message
    finally:
        model.snapshot()
        print 'Stopping threads'
        print 'Done'
