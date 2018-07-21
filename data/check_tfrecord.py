from __future__ import print_function

import tensorflow as tf
import sys
sys.path.insert(0, '../tfmodels')
import argparse
import tfmodels

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def check_record(args):
    
    with tf.Session(config=config) as sess:
        dataset = tfmodels.TFRecordImageMask(
            training_record = args.record_path,
            sess = sess,
            crop_size = args.crop_size,
            ratio = args.ratio,
            batch_size = None,
            prefetch = None,
            shuffle_buffer = 128,
            n_classes = args.classes,
            preprocess = [],
            repeat = False, 
            n_threads = args.threads)

        dataset.print_info()

        idx = 0
        while True:
            try:
                x, y = sess.run([dataset.image_op, dataset.mask_op])
                idx += 1
            except tf.errors.OutOfRangeError:
                print('Reached end of {} examples'.format(idx))
                break

    # idx = 0
    # for example in tf.python_io.tf_record_iterator(args.record_path):
    #     result = tf.train.Example.FromString(example)
    #     idx += 1

    # print(idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('record_path')
    parser.add_argument('--classes', default=5, type=int)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ratio', default=1.0, type=float)
    parser.add_argument('--threads', default=4, type=int)

    args = parser.parse_args()

    check_record(args)