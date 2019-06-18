# https://stackoverflow.com/ \
# questions/33759623/tensorflow-how-to-save-restore-a-model/47235448#47235448

from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import cv2
import sys
import glob
import time
import shutil
import argparse
from sklearn.metrics import (confusion_matrix, f1_score, classification_report)
sys.path.insert(0, '..')
from svs_reader import Slide

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

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


def class_image_list(img_dir, class_num):
    class_image_path = os.path.join(img_dir, '{}'.format(class_num), '*.jpg')
    class_image_list = sorted(glob.glob(class_image_path))
    return class_image_list


def test_list(class_images, sess, image_op, predict_op):
    input_size = image_op.get_shape().as_list()
    x_size, y_size = input_size[1:3]

    def load_image(img_path):
        img = cv2.imread(img_path, -1)[:,:,::-1]
        img = cv2.resize(img, dsize=(x_size, y_size))
        img = img * (1/255.)
        img = np.expand_dims(img, 0)
        return img

    prediction = [sess.run(predict_op, {image_op: load_image(img)}) for img in class_images]
    prediction = np.asarray(prediction)
    print(prediction)
    return np.argmax(prediction, axis=-1)

NUM_CLASSES = 5
def main(img_dir, sess, image_op, predict_op):
    image_lists = {x: class_image_list(img_dir, x) for x in range(NUM_CLASSES)}
    class_predictions = {}
    class_accuracy = {}
    y_trues = {}
    for class_num, class_images in image_lists.items():
        n_images = float(len(class_images))
        y_true = np.asarray([class_num]*len(class_images))
        y_trues[class_num] = y_true
        predictions = test_list(class_images, sess, image_op, predict_op)
        class_predictions[class_num] = predictions
        class_accuracy[class_num] = (predictions == class_num).sum() / n_images
        print(class_num, class_accuracy[class_num])

    return class_predictions, y_trues


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='snapshots/inception_v3')
    parser.add_argument('--val_dir', default='../data/tfhub_val')
    parser.add_argument('--out_file', default='result.tsv')

    args = parser.parse_args()
    model_path = args.model_path
    val_dir = args.val_dir
    out_file = args.out_file
    print(args)

    print(val_dir)
    with tf.Session(config=config) as sess:

        image_op , predict_op = get_input_output_ops(sess, model_path)

        predictions, y_trues = main(val_dir, sess, image_op, predict_op)

        y_predict = [predictions[class_num] for class_num in predictions.keys()]
        y_true = [y_trues[class_num] for class_num in y_trues.keys()]
        y_predict = np.concatenate(y_predict, axis=0); print('y_predict:', y_predict.shape)
        y_true = np.concatenate(y_true, axis=0); print('y_true:', y_true.shape)

        print(confusion_matrix(y_true, y_predict))
        print('\n\n')
        print(classification_report(y_true, y_predict))

        overall_F1 = f1_score(y_true, y_predict, average='macro')
        print('\n\nOVERALL_F1:', overall_F1)
