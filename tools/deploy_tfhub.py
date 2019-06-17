#!/usr/bin/env python

from svsutils import TensorflowIterator, Slide 

import argparse
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# We need this for tfhub models -- consider rolling into svsutils
def get_input_output_ops(sess, model_path):
  input_key = 'image'
  output_key = 'prediction'
  print('loading model {}'.format(model_path))
  signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  meta_graph_def = tf.saved_model.loader.load(
    sess,
    [tf.saved_model.tag_constants.SERVING],
    model_path )
  signature = meta_graph_def.signature_def

  print('getting tensor names:')
  image_tensor_name = signature[signature_key].inputs[input_key].name
  print('input tensor: ', image_tensor_name)
  predict_tensor_name = signature[signature_key].outputs[output_key].name
  print('output tensor:', predict_tensor_name)

  image_op = sess.graph.get_tensor_by_name(image_tensor_name)
  predict_op = sess.graph.get_tensor_by_name(predict_tensor_name)
  print('input:', image_op.get_shape())
  print('output:', predict_op.get_shape())
  return image_op, predict_op


def compute_fn(slide, args, predict_op=None, image_op=None):
  print('Slide with {} tiles'.format(len(slide.tile_list)))
  it_factory = TensorflowIterator(slide, args)
  tfiterator = it_factory.make_iterator()
  img, idx = tfiterator.get_next()

  n = 0
  while True:
    try:
      tile, idx_ = sess.run([img, idx])
      output = sess.run(predict_op, {image_op: tile})
      print(tile.shape, idx_.shape, output.shape)
      print(output)
      slide.place_batch(output, idx_, 'prob', mode='tile')
      if n % 50 == 0:
        print('Batch {}'.format(n))
    except tf.errors.OutOfRangeError:
      print('Finished')
      break
    # except Exception as e:
    #   print('Error caught')
    #   print(e)
    #   break

  # No need to 'make_output' in tile mode
  ret = slide.output_imgs['prob']
  return ret

def main(args, sess):
  with open(args.slides, 'r') as f:
    srclist = [x.strip() for x in f]

  # image_op = tf.placeholder(tf.float32, (args.batchsize, args.process_size,
  #                           args.process_size, 3))
  # module = hub.Module(args.snapshot)
  # predict_op = module(image_op)

  image_op, predict_op = get_input_output_ops(sess, args.snapshot)

  for src in srclist:
    dst = os.path.splitext(src)[0] + '.{}'.format(args.ext)
    dst_base = os.path.basename(dst)
    dst = os.path.join(args.dest, dst_base)
    print(dst)

    slide = Slide(src, args)
    slide.initialize_output('prob', 4, mode='tile',
                            compute_fn=compute_fn)

    ret = slide.compute('prob', args, 
                        predict_op=predict_op, 
                        image_op=image_op)

    np.save(dst, ret)

if __name__ == '__main__':
  """
  standard __name__ == __main__ ?? 
  how to make this nicer
  """
  p = argparse.ArgumentParser()
  # positional arguments for this program
  p.add_argument('slides') 
  p.add_argument('dest') 
  p.add_argument('ext') 
  p.add_argument('snapshot') 


  p.add_argument('--n_classes',  default=4, type=int)

  p.add_argument('-t', default='img.jpg')
  p.add_argument('-b', dest='batchsize', default=4, type=int)
  p.add_argument('-r', dest='ramdisk', default='./', type=str)

  # Slide options - standard
  p.add_argument('--mag',   dest='process_mag', default=10, type=int)
  p.add_argument('--chunk', dest='process_size', default=224, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.5, type=float)

  # Functionals for various parts of the pipeline
  p.add_argument('--function1',   dest='preprocess_fn', 
    default = lambda x: (x / 255.).astype(np.float32)
  )
  p.add_argument('--function3',   dest='normalize_fn', 
    default = lambda x: x
  )

  args = p.parse_args()

  with tf.Session() as sess:
    main(args, sess)