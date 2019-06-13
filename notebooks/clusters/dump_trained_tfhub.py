#!/usr/bin/env python

from __future__ import print_function
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import glob

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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

OUTDIR = 'tsne/inception_v3'
IMGLIST = 'imglist.txt'
MODULE_PATH = '../../tfhub/snapshots/inception_v3'
image_in, predict_op = get_input_output_ops(sess, MODULE_PATH)
_, height, width, _ = image_in.get_shape()

with open(IMGLIST, 'r') as f:
  jpg_list = [L.strip() for L in f]
print(len(jpg_list))

samples = 5
resize = 0.5

y_vectors = []
for imgpath in jpg_list:
    img = cv2.imread(imgpath)[:,:,::-1] / 255.
    img = np.expand_dims(img, axis=0)
    yhat = sess.run(predict_op, feed_dict={image_in: img.astype(np.float32)})
    y_vectors.append(yhat)
  
y_vectors = np.argmax(np.concatenate(y_vectors, axis=0), axis=-1)

y_path = os.path.join(OUTDIR, 'ypred.npy')
print('y_path:', y_path)
print('y_vectors:', y_vectors.shape)
np.save(y_path, y_vectors)
