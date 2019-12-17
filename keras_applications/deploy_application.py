#!/usr/bin/env python

from svsutils import PythonIterator, TensorflowIterator, Slide 

import os
import shutil
import argparse
import traceback
import numpy as np
import tensorflow as tf

# import tensorflow_hub as hub

# We need this for tfhub models -- consider rolling into svsutils
# def get_input_output_ops(sess, model_path):
#   input_key = 'image'
#   output_key = 'prediction'
#   print('loading model {}'.format(model_path))
#   signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
#   meta_graph_def = tf.saved_model.loader.load(
#     sess,
#     [tf.saved_model.tag_constants.SERVING],
#     model_path )
#   signature = meta_graph_def.signature_def

#   print('getting tensor names:')
#   image_tensor_name = signature[signature_key].inputs[input_key].name
#   print('input tensor: ', image_tensor_name)
#   predict_tensor_name = signature[signature_key].outputs[output_key].name
#   print('output tensor:', predict_tensor_name)

#   image_op = sess.graph.get_tensor_by_name(image_tensor_name)
#   predict_op = sess.graph.get_tensor_by_name(predict_tensor_name)
#   print('input:', image_op.get_shape())
#   print('output:', predict_op.get_shape())
#   return image_op, predict_op

def cp_ramdisk(src, ramdisk = '/dev/shm'):
  base = os.path.basename(src)
  dst = os.path.join(ramdisk, base)
  shutil.copyfile(src, dst)
  return dst

# def compute_fn(slide, args, predict_op=None, image_op=None):
def compute_fn(slide, args, model):
  print('Slide with {} tiles'.format(len(slide.tile_list)))
  # it_factory = TensorflowIterator(slide, args)
  it_factory = PythonIterator(slide, args)
  # dataset = it_factory.make_dataset()
  # img, idx = tfiterator.get_next()

  n = 0
  # for tiles, idx_ in dataset:
  for tiles, idx_ in it_factory.yield_batch():
    output = model(tiles, training=False).numpy()
    # print(tiles.shape, idx_.shape, output.shape)
    # print(output)
    slide.place_batch(output, idx_, 'prob', mode='tile')
    n += 1
    if n % 100 == 0:
      print(f'Batch {n} {n*args.batch} tiles')
    # except Exception as e:
    #   print('Error caught')
    #   print(e)
    #   break

  # No need to 'make_output' in tile mode
  ret = slide.output_imgs['prob']
  return ret



def main(args):

  print(f'Load model from {args.snapshot}')
  model = tf.keras.models.load_model(args.snapshot)

  print(f'Processing {args.slide}')
  dst_base = os.path.basename(args.slide).replace('svs', 'npy')

  if not os.path.isdir(args.dest):
    os.makedirs(args.dest)

  dst = os.path.join(args.dest, dst_base)
  print(f'Destination {dst}')

  ramdisk_file = cp_ramdisk(args.slide)
  slide = Slide(ramdisk_file, args)
  try:
    slide.initialize_output('prob', 4, mode='tile', compute_fn=compute_fn)
    ret = slide.compute('prob', args, model=model)
    np.save(dst, ret)

  except Exception as e:
    print('Caught error processing')
    traceback.print_tb(e.__traceback__)
    print(e)

  finally:
    os.remove(ramdisk_file)


if __name__ == '__main__':
  """
  standard __name__ == __main__ ?? 
  how to make this nicer
  """
  p = argparse.ArgumentParser()
  # positional arguments for this program
  p.add_argument('slide') 
  p.add_argument('dest') 
  # p.add_argument('ext') 
  p.add_argument('snapshot') 

  p.add_argument('--n_classes',  default=4, type=int)

  # p.add_argument('-t', default='img.jpg')
  p.add_argument('-b', dest='batch', default=12, type=int)
  p.add_argument('-r', dest='ramdisk', default='/dev/shm', type=str)

  # Slide options - standard
  p.add_argument('--mag',   dest='process_mag', default=10, type=int)
  p.add_argument('--chunk', dest='process_size', default=128, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.25, type=float)

  # Functionals for various parts of the pipeline
  p.add_argument('--function1',   dest='preprocess_fn', 
    default = lambda x: (x / 255.).astype(np.float32)
  )
  p.add_argument('--function3',   dest='normalize_fn', 
    default = lambda x: x
  )

  args = p.parse_args()

  # with tf.Session() as sess:
  main(args)