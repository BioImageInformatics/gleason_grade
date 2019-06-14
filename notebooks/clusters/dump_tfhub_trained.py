#!/usr/bin/env python

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import sys
import os
import argparse

from dump_imgs import list_images

def get_input_output_ops(sess, model_path):
  input_key = 'image'
  output_key = 'prediction'
  print('Loading model {}'.format(model_path))
  signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  meta_graph_def = tf.saved_model.loader.load(
      sess,
      [tf.saved_model.tag_constants.SERVING],
      model_path )
  signature = meta_graph_def.signature_def

  print('Getting tensor names:')
  image_tensor_name = signature[signature_key].inputs[input_key].name
  print('Input tensor: ', image_tensor_name)
  predict_tensor_name = signature[signature_key].outputs[output_key].name
  print('Output tensor:', predict_tensor_name)

  image_op = sess.graph.get_tensor_by_name(image_tensor_name)
  predict_op = sess.graph.get_tensor_by_name(predict_tensor_name)
  print('Input:', image_op.get_shape())
  print('Output:', predict_op.get_shape())
  return image_op, predict_op


def main(args, sess):

  image_in, predict_op = get_input_output_ops(sess, args.snapshot)
  yvect = []
  imglist = list_images(args.source, ext='jpg')
  for idx, imgpath in enumerate(imglist):
    img = cv2.imread(imgpath)[:,:,::-1]
    img = np.expand_dims(img / 255., 0)

    yhat = sess.run(predict_op, feed_dict={image_in: img})
    ymax = np.argmax(yhat)
    if args.collapse_classes:
      if ymax > 1:
        ymax -= 1
    yvect.append(ymax)

    if idx % 100 == 0:
      print('{} --> {} --> {} --> {}'.format(imgpath, img.shape, yhat.shape, ymax))

  yvect = np.asarray(yvect)
  dst = os.path.join(args.dest, 'ypred.npy')
  np.save(dst, yvect)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('source')
  parser.add_argument('dest')
  parser.add_argument('snapshot')
  parser.add_argument('--collapse_classes', default=False, action='store_true')

  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    main(args, sess)
