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

class FCN(Segmentation):
    fcn_defaults={
        'k_size': [7, 5, 3, 3],
        'conv_kernels': [64, 128, 256, 256],
        'name': 'fcn',
        'snapshot_name': 'fcn'}

    def __init__(self, **kwargs):
        self.fcn_defaults.update(**kwargs)
        super(FCN, self).__init__(**self.fcn_defaults)

        assert self.n_classes is not None
        assert self.conv_kernels is not None


    ## Layer flow copied from:
    ## https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py
    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print 'FCN Model'
        k_size = self.k_size
        nonlin = self.nonlin
        print 'Non-linearity:', nonlin

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print '\t x_in', x_in.get_shape()

            c0_0 = nonlin(conv(x_in, self.conv_kernels[0], k_size=k_size[0], stride=1, var_scope='c0_0'))
            c0_1 = nonlin(conv(c0_0, self.conv_kernels[0], k_size=k_size[0], stride=1, var_scope='c0_1'))
            c0_pool = tf.nn.max_pool(c0_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c0_pool')
            print '\t c0_pool', c0_pool.get_shape() ## 128

            c1_0 = nonlin(conv(c0_pool, self.conv_kernels[1], k_size=k_size[1], stride=1, var_scope='c1_0'))
            c1_1 = nonlin(conv(c1_0, self.conv_kernels[1], k_size=k_size[1], stride=1, var_scope='c1_1'))
            c1_pool = tf.nn.max_pool(c1_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c1_pool')
            print '\t c1_pool', c1_pool.get_shape() ## 64

            c2_0 = nonlin(conv(c1_pool, self.conv_kernels[2], k_size=k_size[2], stride=1, var_scope='c2_0'))
            c2_1 = nonlin(conv(c2_0, self.conv_kernels[2], k_size=k_size[2], stride=1, var_scope='c2_1'))
            c2_1 = tf.contrib.nn.alpha_dropout(c2_1, keep_prob=keep_prob)
            c2_pool = tf.nn.max_pool(c2_1, [1,4,4,1], [1,4,4,1], padding='VALID',
                name='c2_pool')
            print '\t c2_pool', c2_pool.get_shape() ## 32

            c3_0 = nonlin(conv(c2_pool, self.conv_kernels[3], k_size=k_size[3], stride=1, var_scope='c3_0'))
            c3_1 = nonlin(conv(c3_0, self.conv_kernels[3], k_size=k_size[3], stride=1, var_scope='c3_1'))
            c3_1 = tf.contrib.nn.alpha_dropout(c3_1, keep_prob=keep_prob)
            c3_pool = tf.nn.max_pool(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID',
                name='c3_pool')
            print '\t c3_pool', c3_pool.get_shape()  ## inputs / 16 = 16

            ## The actual architecture has one more level
            # c4_0 = nonlin(conv(c3_pool, self.conv_kernels[4], k_size=3, stride=1, var_scope='c4_0'))
            # c4_0 = tf.contrib.nn.alpha_dropout(c4_0, keep_prob=keep_prob)
            # c4_1 = nonlin(conv(c4_0, self.conv_kernels[4], k_size=3, stride=1, var_scope='c4_1'))
            # c4_pool = tf.nn.max_pool(c3_1, [1,2,2,1], [1,2,2,1], padding='VALID',
            #     name='c4_pool')
            # print '\t c4_pool', c4_pool.get_shape()  ## inputs / 32 = 8

            ## Type 1 - much simpler
            # upscore3 = nonlin(deconv(c3_pool, self.n_classes, k_size=36, upsample_rate=32, var_scope='ups3'))
            # upscore2 = nonlin(deconv(c2_pool, self.n_classes, k_size=18, upsample_rate=16, var_scope='ups2'))
            # upscore1 = nonlin(deconv(c1_pool, self.n_classes, k_size=7, upsample_rate=4, var_scope='ups1'))
            # print '\t upscore3', upscore3.get_shape()
            # print '\t upscore2', upscore2.get_shape()
            # print '\t upscore1', upscore1.get_shape()
            #
            # upscore_concat = tf.concat([upscore3, upscore2, upscore1], axis=-1)
            # print '\t upscore_concat', upscore_concat.get_shape()
            # preout = nonlin(conv(upscore_concat, self.deconv_kernels[0], k_size=5, stride=1, var_scope='preout'))
            # print '\t preout', preout.get_shape()

            ## Type 2
            prediction_3 = nonlin(conv(c3_pool, self.n_classes, stride=1, var_scope='pred3'))
            prediction_2 = nonlin(conv(c2_pool, self.n_classes, stride=1, var_scope='pred2'))
            prediction_1 = nonlin(conv(c1_pool, self.n_classes, stride=1, var_scope='pred1'))
            print '\t prediction_3', prediction_3.get_shape()
            print '\t prediction_2', prediction_2.get_shape()
            print '\t prediction_1', prediction_1.get_shape()

            upscore3 = nonlin(deconv(prediction_3, self.n_classes, k_size=4, upsample_rate=2, var_scope='ups3'))
            print '\t upscore3', upscore3.get_shape()
            upscore3 = upscore3 + prediction_2
            upscore3_ups = nonlin(deconv(upscore3, self.n_classes, k_size=4, upsample_rate=2, var_scope='ups3_ups'))
            print '\t upscore3_ups', upscore3_ups.get_shape()

            upscore2 = nonlin(deconv(prediction_2, self.n_classes, k_size=4, upsample_rate=2, var_scope='ups2'))
            print '\t upscore2', upscore2.get_shape()
            upscore2 = upscore2 + upscore3_ups
            upscore2_ups = nonlin(deconv(upscore2, self.n_classes, k_size=4, upsample_rate=2, var_scope='ups2_ups'))
            print '\t upscore2_ups', upscore2_ups.get_shape()
            upscore2 = prediction_1 + upscore2_ups

            preout = nonlin(deconv(upscore2, self.n_classes, k_size=4, upsample_rate=4, var_scope='preout'))
            print '\t preout', preout.get_shape()

            y_hat = conv(preout, self.n_classes, k_size=3, stride=1, var_scope='y_hat')
            print '\t y_hat', y_hat.get_shape()

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
