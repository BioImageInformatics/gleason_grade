from __future__ import print_function
import tensorflow as tf
import sys

sys.path.insert(0, '../tfmodels')
from tfmodels import Segmentation
from tfmodels import ( batch_norm,
    conv,
    conv_cond_concat,
    deconv,
    linear,
    lrelu)

"""
Fully Convolutional Networks

@inproceedings{long2015fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3431--3440},
  year={2015}
}

"""

class FCN(Segmentation):
    fcn_defaults={
        'k_size': [3, 3, 3, 3],
        # 'conv_kernels': [64, 128, 256, 256, 512], ## Original dimensions
        'conv_kernels': [32, 32, 64, 128, 256], ## Reduced dimensions by half
        'fc_dim': 1024,
        'use_optimizer': 'Adam',
        'name': 'fcn',
        'n_classes': 5,
        'snapshot_name': 'fcn'}

    def __init__(self, **kwargs):
        self.fcn_defaults.update(**kwargs)
        super(FCN, self).__init__(**self.fcn_defaults)
        assert self.n_classes is not None
        assert self.conv_kernels is not None


    ## Layer flow copied from:
    ## https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py
    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print('FCN Model')
        k_size = self.k_size
        nonlin = self.nonlin
        print('Non-linearity:', nonlin)

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print('\t x_in', x_in.get_shape())

            c0_0 = nonlin(conv(x_in, self.conv_kernels[0], k_size=k_size[0], stride=1, var_scope='c0_0'))
            c0_pool = tf.nn.max_pool(c0_0, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c0_pool')
            print('\t c0_pool', c0_pool.get_shape()) ## in / 2
            self.conv1 = tf.identity(c0_pool)

            c1_0 = nonlin(conv(c0_pool, self.conv_kernels[1], k_size=k_size[1], stride=1, var_scope='c1_0'))
            c1_pool = tf.nn.max_pool(c1_0, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c1_pool')
            print('\t c1_pool', c1_pool.get_shape())## in / 4

            c2_0 = nonlin(conv(c1_pool, self.conv_kernels[2], k_size=k_size[2], stride=1, var_scope='c2_0'))
            c2_pool = tf.nn.max_pool(c2_0, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c2_pool')
            print('\t c2_pool', c2_pool.get_shape())## in / 8

            c3_0 = nonlin(conv(c2_pool, self.conv_kernels[3], k_size=k_size[3], stride=1, var_scope='c3_0'))
            c3_pool = tf.nn.max_pool(c3_0, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c3_pool')
            print('\t c3_pool', c3_pool.get_shape())  ## in / 32

            c4_0 = nonlin(conv(c3_pool, self.conv_kernels[4], k_size=3, stride=1, var_scope='c4_0'))
            c4_pool = tf.nn.max_pool(c4_0, [1,4,4,1], [1,4,4,1], padding='VALID',
                name='c4_pool')
            print('\t c4_pool', c4_pool.get_shape())  ## in / 64

            ## Pull out the shape of c4_pool so the result is ? x 1 x 1 x self.fc_dim
            c4_shape = c4_pool.get_shape()
            kern_size = c4_shape[1].value
            fc_1 = nonlin(conv(c4_pool, self.fc_dim, k_size=kern_size, stride=kern_size, var_scope='fc_1'))
            fc_1 = tf.contrib.nn.alpha_dropout(fc_1, keep_prob=keep_prob)
            print('\t fc_1', fc_1.get_shape())  ##
            self.bottleneck = tf.identity(fc_1)

            fc_2 = nonlin(conv(fc_1, self.fc_dim, k_size=1, stride=1, var_scope='fc_2'))
            fc_2 = tf.contrib.nn.alpha_dropout(fc_2, keep_prob=keep_prob)
            print('\t fc_2', fc_2.get_shape())  ##

            score_fr = conv(fc_2, self.n_classes, stride=1, var_scope='score_fr')
            print('\t score_fr', score_fr.get_shape())  ##

            ## Upscoring
            # prediction_4 = nonlin(conv(c4_pool, self.n_classes, stride=1, var_scope='pred4'))
            prediction_3 = nonlin(conv(c3_pool, self.n_classes, stride=1, var_scope='pred3'))
            prediction_2 = nonlin(conv(c2_pool, self.n_classes, stride=1, var_scope='pred2'))
            prediction_1 = nonlin(conv(c1_pool, self.n_classes, stride=1, var_scope='pred1'))
            prediction_0 = nonlin(conv(c0_pool, self.n_classes, stride=1, var_scope='pred0'))
            print('\t prediction_3', prediction_3.get_shape())
            print('\t prediction_2', prediction_2.get_shape())
            print('\t prediction_1', prediction_1.get_shape())
            print('\t prediction_0', prediction_0.get_shape())

            ## Get back the proper dimensions from the downsampling branch
            shape_up = prediction_3.get_shape()
            ups = shape_up[1].value
            upscore3 = nonlin(deconv(score_fr, self.n_classes, k_size=4, upsample_rate=ups, var_scope='upscore3',
                shape=tf.shape(prediction_3)))
            print('\t upscore3', upscore3.get_shape())
            upscore3_fuse = upscore3 + prediction_3
            print('\t upscore3_fuse', upscore3_fuse.get_shape())

            upscore2 = nonlin(deconv(upscore3_fuse, self.n_classes, k_size=4, upsample_rate=2, var_scope='upscore2',
                shape=tf.shape(prediction_2)))
            print('\t upscore2', upscore2.get_shape())
            upscore2_fuse = upscore2 + prediction_2
            print('\t upscore2_fuse', upscore2_fuse.get_shape())

            upscore1 = nonlin(deconv(upscore2_fuse, self.n_classes, k_size=4, upsample_rate=2, var_scope='upscore1',
                shape=tf.shape(prediction_1)))
            print('\t upscore1', upscore1.get_shape())
            upscore1_fuse = upscore1 + prediction_1
            print('\t upscore1_fuse', upscore1_fuse.get_shape())

            upscore0 = nonlin(deconv(upscore1_fuse, self.n_classes, k_size=4, var_scope='upscore0'))
            print('\t upscore0', upscore0.get_shape())
            upscore0_fuse = upscore0 + prediction_0
            print('\t upscore0_fuse', upscore0_fuse.get_shape())

            y_hat = deconv(upscore0_fuse, self.n_classes, k_size=4, var_scope='y_hat')
            print('\t y_hat', y_hat.get_shape())

            self.intermediate_ops = {
                '01.c0_pool': c0_pool,
                '02.c1_pool': c1_pool,
                '03.c2_pool': c2_pool,
                '04.c3_pool': c3_pool,
                '05.c4_pool': c4_pool,
                '06.prediction_0': prediction_0,
                '07.prediction_1': prediction_1,
                '08.prediction_2': prediction_2,
                '09.prediction_3': prediction_3,
                '10.upscore3': upscore3,
                '11.upscore2': upscore2,
                '12.upscore1': upscore1,
                '13.upscore0': upscore0,
                '14.y_hat': y_hat
                }
            return y_hat

    ## Overload to fill in the default keep_prob
    def train_step(self, lr):
        self.global_step += 1
        fd = {self.keep_prob: 0.5,
              self.training: True,
              self.learning_rate: lr}
        self.sess.run(self.seg_training_op_list, feed_dict=fd)

        if self.global_step % self.summary_iters == 0:
            self._write_scalar_summaries(lr)

        if self.global_step % self.summary_image_iters == 0:
            self._write_image_summaries()


class Training(FCN):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(Training, self).__init__(**self.train_defaults)


class Inference(FCN):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(Inference, self).__init__(**self.inference_defaults)
