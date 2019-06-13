from __future__ import print_function
import tensorflow as tf
import sys

from tfmodels import Segmentation
from tfmodels.utilities.ops import *

"""
Original DenseNet: https://arxiv.org/abs/1608.06993

@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q and van der Maaten, Laurens},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  volume={1},
  number={2},
  pages={3},
  year={2017}
}

Fully Convolutional DenseNet: https://arxiv.org/abs/1611.09326

@inproceedings{jegou2017one,
  title={The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation},
  author={J{\'e}gou, Simon and Drozdzal, Michal and Vazquez, David and Romero, Adriana and Bengio, Yoshua},
  booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on},
  pages={1175--1183},
  year={2017},
  organization={IEEE}
}

"""
class DenseNet(Segmentation):
    densenet_defaults={
        ## Ad-hoc
        # 'class_weights': [0.9, 0.9, 0.9, 0.3, 0.3],
        ## Number of layers to use for each dense block
        'dense_stacks': [4, 5, 7, 10],
        ## The parameter k in the paper. Dense blocks end up with L*k kernels
        'growth_rate': 48,
        ## Kernel size for all layers. either scalar or list same len as dense_stacks
        'k_size': 3,
        'n_classes': 5,
        'name': 'densenet',
    }

    def __init__(self, **kwargs):
        self.densenet_defaults.update(**kwargs)

        ## not sure sure it's good to do this first
        for key, val in self.densenet_defaults.items():
            setattr(self, key, val)

        ## The smallest dimension after all downsampling has to be >= 1
        self.n_dense = len(self.dense_stacks)
        print('Requesting {} dense blocks'.format(self.n_dense))
        start_size = min(self.x_dims[:2])
        min_dimension = start_size / np.power(2,self.n_dense+1)
        print('MINIMIUM DIMENSION: ', min_dimension)
        assert min_dimension >= 1

        super(DenseNet, self).__init__(**self.densenet_defaults)

        ## Check input shape is compatible with the number of downsampling modules


    """
    Dense blocks do not change h or w.

    Define the normal CNN transformation as x_i = H(x_(i-1)) .
    DenseNet uses an iterative concatenation for all layers:
        x_i = H([x_(i-1), x_(i-2), ..., x_0])

    Given x_0 ~ (batch_size, h, w, k_in)
    Return x_i ~ (batch_size, h, w, k_in + stacks*growth_rate)
    """
    def _dense_block(self, x_flow, n_layers, concat_input=True, keep_prob=0.8, block_num=0, name_scope='dense'):
        nonlin = self.nonlin
        conv_settings = {'n_kernel': self.growth_rate, 'stride': 1, 'k_size': self.k_size, 'no_bias': 0}
        conv_settings_b = {'n_kernel': self.growth_rate*4, 'stride': 1, 'k_size': 1, 'no_bias': 0}
        print('Dense block #{} ({})'.format(block_num, name_scope))

        concat_list = [x_flow]
        # print('\t x_flow', x_flow.get_shape())
        with tf.variable_scope('{}_{}'.format(name_scope, block_num)):
            for l_i in range(n_layers):
                layer_name = 'd{}_l{}'.format(block_num, l_i)
                x_b = nonlin(conv(x_flow, var_scope=layer_name+'b', **conv_settings_b))
                x_hidden = nonlin(conv(x_b, var_scope=layer_name, **conv_settings))
                #x_hidden = tf.contrib.nn.alpha_dropout(x_hidden, keep_prob=keep_prob)
                concat_list.append(x_hidden)
                x_flow = tf.concat(concat_list, axis=-1, name='concat'+layer_name)
                # print('\t\t CONCAT {}:'.format(block_num, l_i), x_flow.get_shape())

            if concat_input:
                x_i = tf.concat(concat_list, axis=-1, name='concat_out')
            else:
                x_i = tf.concat(concat_list[1:], axis=-1, name='concat_out')

        return x_i


    ## theta is the compression factor, 0 < theta <= 1
    ## If theta = 1, then k_in = k_out
    def _transition_down(self, x_in, td_num, theta=0.5, keep_prob=0.8, name_scope='td'):
        nonlin = self.nonlin
        k_out = int(x_in.get_shape().as_list()[-1] * theta)
        print('\t Transition Down with k_out=', k_out)
        conv_settings = {'n_kernel': k_out, 'stride': 1, 'k_size': 1, 'no_bias': 0}

        with tf.variable_scope('{}_{}'.format(name_scope, td_num)):
            x_conv = nonlin(conv(x_in, var_scope='conv', **conv_settings))
            x_conv = tf.contrib.nn.alpha_dropout(x_conv, keep_prob=keep_prob)
            x_pool = tf.nn.max_pool(x_conv, [1,2,2,1], [1,2,2,1], padding='VALID', name='pool')

        return x_pool


    def _transition_up(self, x_in, tu_num, theta=0.5, keep_prob=0.8, name_scope='tu'):
        k_out = int(x_in.get_shape().as_list()[-1] * theta)
        print('\t Transition Up with k_out=', k_out)
        deconv_settings = {'n_kernel': k_out, 'upsample_rate': 2, 'k_size': 3, 'no_bias': 0}

        with tf.variable_scope('{}_{}'.format(name_scope, tu_num)):
            x_deconv = deconv(x_in, var_scope='TU', **deconv_settings)
            x_deconv = tf.contrib.nn.alpha_dropout(x_deconv, keep_prob=keep_prob)

        return x_deconv


    """
    x_in is (batch_size, h, w, channels)

    Similar to Table 2 in https://arxiv.org/abs/1611.09326
    """
    def model(self, x_in, keep_prob=0.8, reuse=False, training=True):
        print('DenseNet Model')
        nonlin = self.nonlin
        print('Non-linearity:', nonlin)

        self.intermediate_ops = {}; op_idx = 1
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            print('\t x_in', x_in.get_shape())

            ## First convolution gets the ball rolling with a pretty big filter
            dense_ = nonlin(conv(x_in, n_kernel=self.growth_rate*2, stride=2, k_size=5, var_scope='conv1'))
            dense_ = tf.nn.max_pool(dense_, [1,2,2,1], [1,2,2,1], padding='VALID', name='c1_pool')
            self.intermediate_ops['{:02d}. Conv1'.format(op_idx)] = dense_; op_idx += 1

            ## Downsampling path
            self.downsample_list = []
            for i_, n_ in enumerate(self.dense_stacks[:-1]):
                dense_i = self._dense_block(dense_, n_, keep_prob=keep_prob, block_num=i_, name_scope='dd')
                dense_ = tf.concat([dense_i, dense_], axis=-1, name='concat_down_{}'.format(i_))
                self.downsample_list.append(dense_)
                # print('\t DENSE: ', dense_.get_shape())

                dense_ = self._transition_down(dense_, i_, keep_prob=keep_prob)
                self.intermediate_ops['{:02d}. Down{}'.format(op_idx, i_)] = dense_; op_idx += 1

            ## bottleneck dense layer
            dense_ = self._dense_block(dense_, self.dense_stacks[-1],
                keep_prob=keep_prob, block_num=len(self.dense_stacks)-1)
            dense_ = tf.contrib.nn.alpha_dropout(dense_, keep_prob=keep_prob)
            self.intermediate_ops['{:02d}. Bottleneck'.format(op_idx)] = dense_; op_idx += 1

            print('\t Bottleneck: ', dense_.get_shape())

            ## Upsampling path -- concat skip connections each time
            self.upsample_list = []
            for i_, n_ in enumerate(reversed(self.dense_stacks[:-1])):
                dense_ = self._transition_up(dense_, tu_num=i_, keep_prob=keep_prob)

                dense_ = tf.concat([dense_, self.downsample_list[-(i_+1)]],
                    axis=-1, name='concat_skip_{}'.format(i_))

                dense_ = self._dense_block(dense_, n_, concat_input=False,
                    keep_prob=keep_prob, block_num=i_, name_scope='du')
                self.upsample_list.append(dense_)
                self.intermediate_ops['{:02d}. Up{}'.format(op_idx, i_)] = dense_; op_idx += 1

            ## Classifier layer
            y_hat_0 = nonlin(deconv(dense_, n_kernel=self.growth_rate*4, k_size=5, pad='SAME', var_scope='y_hat_0'))
            self.intermediate_ops['{:02d}. y_hat_0'.format(op_idx)] = dense_; op_idx += 1 
            y_hat = deconv(y_hat_0, n_kernel=self.n_classes, k_size=5, pad='SAME', var_scope='y_hat')

        return y_hat

    ## Overload to fill in the default keep_prob
    def train_step(self, lr):
        self.global_step += 1
        fd = {self.keep_prob: 0.8,
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

    # def test_step(self, step_delta, keep_prob=1.0):
    #     fd = {self.keep_prob: keep_prob,
    #           self.training: False}
    #     summary_str, test_loss_ = self.sess.run([self.summary_test_ops, self.loss], feed_dict=fd)
    #     self.summary_writer.add_summary(summary_str, self.global_step+step_delta)
    #     print('#### GLEASON GRADE TEST #### [{:07d}] writing test summaries (loss={:3.3f})'.format(self.global_step, test_loss_))
    #     return test_loss_


class Training(DenseNet):
    train_defaults = { 'mode': 'TRAIN' }

    def __init__(self, **kwargs):
        self.train_defaults.update(**kwargs)
        super(Training, self).__init__(**self.train_defaults)


class Inference(DenseNet):
    inference_defaults = { 'mode': 'TEST' }

    def __init__(self, **kwargs):
        self.inference_defaults.update(**kwargs)
        super(Inference, self).__init__(**self.inference_defaults)
