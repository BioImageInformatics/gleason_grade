#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import cv2
from split_datasets import load_dataset

import argparse

from tensorflow.keras.applications import ResNet50V2, VGG16, VGG19, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


def define_model(architecture, input_shape=(128, 128, 3), n_classes=4):
  if architecture == 'ResNet50V2':
    base_model = ResNet50V2(weights='imagenet', include_top=False,
      input_shape=input_shape)
  elif architecture == 'VGG16':
    base_model = VGG16(weights='imagenet', include_top=False,
      input_shape=input_shape)
  elif architecture == 'VGG19':
    base_model = VGG19(weights='imagenet', include_top=False,
      input_shape=input_shape)
  elif architecture == 'DenseNet121':
    base_model = DenseNet121(weights='imagenet', include_top=False,
      input_shape=input_shape)

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation='relu')(x)
  predictions = Dense(n_classes, activation='softmax')(x)

  model = Model(inputs = base_model.input, outputs=predictions)
  return model


def perturb(x, y):
  x = tf.image.random_flip_left_right(x)
  x = tf.image.random_flip_up_down(x)
  x = tf.image.random_brightness(x, 0.15)
  x = tf.image.random_contrast(x, 0.75, 1.)
  x = tf.image.random_hue(x, 0.15)

  return x, y

def define_dataset(data_x, data_y, args):
  dataset = (tf.data.Dataset.from_tensor_slices((data_x, data_y))
             .repeat(args.epochs)
             .map(perturb, num_parallel_calls=8)
             .prefetch(100)
             .batch(args.batch_size)
  )
  return dataset

# @tf.function
def train(model, dataset, optimizer, args):

  for x, y in dataset:
    with tf.GradientTape() as tape:
      yhat = model(y)
      loss = categorical_crossentropy(y, yhat)
    
    grads = tape.gradient(model.trainable_variables, loss)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def main(args):
  image_home = args.image_home
  mask_home = args.mask_home
  patient_list = [p.strip() for p in open(args.patient_list, 'r')]

  data_x, data_y = load_dataset(image_home, mask_home, patient_list,
    size = args.size, downsample=args.downsample, overlap=args.overlap)

  input_shape = (data_x.shape[1], data_x.shape[2], 3)
  model = define_model(args.architecture, input_shape, args.n_classes)

  # optimizer = Adam(lr=args.learning_rate)
  dataset = define_dataset(data_x, data_y, args)

  for x, y in dataset:
    print(x.shape, y.shape)
    break

  model.compile(optimizer = Adam(lr = args.learning_rate), loss='categorical_crossentropy')
  model.fit(dataset)

  model.save(args.savepath, include_optimizer=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('image_home')
  parser.add_argument('mask_home')
  parser.add_argument('patient_list')
  parser.add_argument('savepath')
  
  parser.add_argument('--architecture', default='ResNet50V2', 
               help = 'One of the named architectures in `tf.keras.applications` \
                      (https://keras.io/applications/)')
  parser.add_argument('--batch_size', default=64, type=int) 
  parser.add_argument('--epochs', default=20, type=int) 
  parser.add_argument('--augmentation', action='store_true') 
  parser.add_argument('--n_classes', default=4, type=int) 
  parser.add_argument('--learning_rate', default=0.0001, type=float) 

  parser.add_argument('--size', default=256, type=int)
  parser.add_argument('--downsample', default=0.5, type=float)
  parser.add_argument('--overlap', default=1.5, type=float)

  args = parser.parse_args()

  main(args)