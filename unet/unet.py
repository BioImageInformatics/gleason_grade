from __future__ import print_function
import tensorflow as tf
import sys

sys.path.insert(0, '../tfmodels')
from tfmodels import Segmentation
from tfmodels.utilities.ops import *

""" Basic U-Net
References:
https://arxiv.org/abs/1505.04597
https://github.com/zhixuhao/unet
https://github.com/jakeret/tf_unet

Replace relu + batch norm with selu
replace dropout with alpha_dropout

Ronneberger et al, (2015):
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
"""
class UNet(Segmentation):

    def __init__(self, **kwargs):
        unet_defaults={
            ## Ad-hoc
            # 'class_weights': [0.9, 0.9, 0.9, 0.3, 0.3],
            'k_size': 3,
            'conv_kernels': [64, 128, 256, 512, 1024],
            'n_classes': 5,
            'x_dim': [256, 256, 3], ## default
            'name': 'unet', }
        unet_defaults.update(**kwargs)

        for key, val in unet_defaults.items():
            setattr(self, key, val)

        super(UNet, self).__init__(**unet_defaults)

    def model(self, x_in, keep_prob=0.5, reuse=False, training=True):
        print('DenseNet Model')
        nonlin = self.nonlin
        print('Non-linearity:', nonlin)

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print('\t x_in', x_in.get_shape())
            conv_opts = {'k_size': 3, 'stride': 1, 'pad': 'SAME'}
            deconv_opts = {'upsample_rate': 2, 'k_size': 2}
            pool_opts = {'ksize': [1,2,2,1], 'strides': [1,2,2,1], 'padding': 'VALID'}
            kerns = self.conv_kernels
            ## x_in = 256, 3
            down1_0 = nonlin(conv(x_in,    n_kernel=kerns[0] ,var_scope='down1_0', **conv_opts))
            down1_1 = nonlin(conv(down1_0, n_kernel=kerns[0] ,var_scope='down1_1', **conv_opts))
            print('down1_1', down1_1.get_shape())
            ## 256, 64

            pool1 = tf.nn.max_pool(down1_1, **pool_opts)
            down2_0 = nonlin(conv(pool1,   n_kernel=kerns[1] ,var_scope='down2_0', **conv_opts))
            down2_1 = nonlin(conv(down2_0, n_kernel=kerns[1] ,var_scope='down2_1', **conv_opts))
            print('down2_1', down2_1.get_shape())
            ## 128, 128

            pool2 = tf.nn.max_pool(down2_1, **pool_opts)
            down3_0 = nonlin(conv(pool2,   n_kernel=kerns[2] ,var_scope='down3_0', **conv_opts))
            down3_1 = nonlin(conv(down3_0, n_kernel=kerns[2] ,var_scope='down3_1', **conv_opts))
            print('down3_1', down3_1.get_shape())
            ## 64, 256

            pool3 = tf.nn.max_pool(down3_1, **pool_opts)
            down4_0 = nonlin(conv(pool3,   n_kernel=kerns[3] ,var_scope='down4_0', **conv_opts))
            down4_1 = nonlin(conv(down4_0, n_kernel=kerns[3] ,var_scope='down4_1', **conv_opts))
            down4_1 = tf.contrib.nn.alpha_dropout(down4_1, keep_prob=keep_prob)
            print('down4_1', down4_1.get_shape())
            ## 32, 512

            pool4 = tf.nn.max_pool(down4_1, **pool_opts)
            down5_0 = nonlin(conv(pool4,   n_kernel=kerns[4] ,var_scope='down5_0', **conv_opts))
            down5_1 = nonlin(conv(down5_0, n_kernel=kerns[4] ,var_scope='down5_1', **conv_opts))
            down5_1 = tf.contrib.nn.alpha_dropout(down5_1, keep_prob=keep_prob)
            print('down5_1', down5_1.get_shape())
            ## 16, 1024

            up4 = nonlin(deconv(down5_1, n_kernel=kerns[3], var_scope='up4', **deconv_opts)) ## 32 x 512
            concat4 = crop_concat(down4_1, up4)
            concat4 = tf.contrib.nn.alpha_dropout(concat4, keep_prob=keep_prob)
            print('concat4', concat4.get_shape())
            up4_1 = nonlin(conv(concat4, n_kernel=kerns[3], var_scope='up4_1', **conv_opts))
            up4_2 = nonlin(conv(up4_1,   n_kernel=kerns[3], var_scope='up4_2', **conv_opts))
            print('up4_2', up4_2.get_shape())

            up3 = nonlin(deconv(up4_2, n_kernel=kerns[2], var_scope='up3', **deconv_opts)) ## 64 x 256
            concat3 = crop_concat(down3_1, up3)
            concat3 = tf.contrib.nn.alpha_dropout(concat3, keep_prob=keep_prob)
            print('concat3', concat3.get_shape())
            up3_1 = nonlin(conv(concat3, n_kernel=kerns[2], var_scope='up3_1', **conv_opts))
            up3_2 = nonlin(conv(up3_1,   n_kernel=kerns[2], var_scope='up3_2', **conv_opts))
            print('up3_2', up3_2.get_shape())

            up2 = nonlin(deconv(up3_2, n_kernel=kerns[1], var_scope='up2', **deconv_opts)) ## 128 x 128
            concat2 = crop_concat(down2_1, up2)
            print('concat2', concat2.get_shape())
            up2_1 = nonlin(conv(concat2, n_kernel=kerns[1], var_scope='up2_1', **conv_opts))
            up2_2 = nonlin(conv(up2_1,   n_kernel=kerns[1], var_scope='up2_2', **conv_opts))
            print('up2_2', up2_2.get_shape())

            up1 = nonlin(deconv(up2_2, n_kernel=kerns[0], var_scope='up1', **deconv_opts)) ## 256 x 64
            concat1 = crop_concat(down1_1, up1)
            print('concat1', concat1.get_shape())
            up1_1 = nonlin(conv(concat1, n_kernel=kerns[0], var_scope='up1_1', **conv_opts))
            up1_2 = nonlin(conv(up1_1,   n_kernel=kerns[0], var_scope='up1_2', **conv_opts))
            print('up1_2', up1_2.get_shape())

            y_hat = conv(up1_2, n_kernel=self.n_classes, var_scope='y_hat', **conv_opts)
            print('y_hat', y_hat.get_shape())

            self.intermediate_ops = {
                '01. Down1': down1_1,
                '02. Down2': down2_1,
                '03. Down3': down3_1,
                '04. Down4': down4_1,
                '05. Up4': up4_2,
                '06. Up3': up3_2,
                '07. Up2': up2_2,
                '08. Up1': up1_2,
                '09. y_hat': y_hat,
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

    def _make_training_ops(self):
        with tf.name_scope('segmentation_losses'):
            self._make_segmentation_loss()

            ## Unused except in pretraining or specificially requested
            self.seg_training_op = self.optimizer.minimize(
                self.seg_loss, var_list=self.var_list, name='{}_seg_train'.format(self.name))

            self.seg_loss_sum = tf.summary.scalar('seg_loss', self.seg_loss)
            self.summary_op_list.append(self.seg_loss_sum)

            self.loss = self.seg_loss

            self.batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.batch_norm_updates):
                self.train_op = self.optimizer.minimize(self.loss,
                    var_list=self.var_list, name='{}_train'.format(self.name))

            print('Setting up batch norm update ops')
            # self.batch_norm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # self.seg_training_op_list.append(self.batch_norm_updates)
            self.seg_training_op_list.append(self.train_op)


class Training(UNet):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(Training, self).__init__(**self.train_defaults)


class Inference(UNet):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(Inference, self).__init__(**self.inference_defaults)
