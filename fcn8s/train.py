import numpy as np
import tensorflow as tf
import sys, datetime, os, time

sys.path.insert(0, '../tfmodels')
import tfmodels

sys.path.insert(0, '.')
from fcn8s import Training

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_record_path = '../data/gleason_grade_train.tfrecord'
test_record_path =  '../data/gleason_grade_val.tfrecord'
n_classes = 5
batch_size = 4
crop_size = 512
image_ratio = 0.25
x_dims = [int(crop_size*image_ratio),
          int(crop_size*image_ratio),
          3]

epochs = 500
iterations = 500/batch_size
snapshot_epochs = 10
step_start = 0

prefetch = 512
threads = 8

basedir = '5x'
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
        n_classes = n_classes,
        log_dir = log_dir,
        save_dir = save_dir,
        summary_iters = 10,
        summary_image_iters = iterations,
        summary_image_n = 4,
        # summarize_grads = True,
        # summarize_vars = True,
        x_dims = x_dims)
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
