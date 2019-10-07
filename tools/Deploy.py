#!/usr/bin/env python
"""
This example deploys a classifier to a list of SVS slides

Utilities demonstrated here:

cpramdisk      - manages and copies data between slow and fast media
Slide          - core object for managing slide data read/write
PythonIterator - hooks for creating generators out of a Slide
xx
TensorflowIterator - A wrapped PythonIterator with multithreading
                     and direct integration with TensorFlow graphs

This script takes advantage of model constructors defined in 
https://github.com/nathanin/milk


Usage
-----
```
python Example_classifier.py [slides.txt] [model/snapshot.h5] [encoder type] [options]
```
These are ending up being drop-in methods
for deploy scripts in other applications


June 2019
"""
from svsutils import repext
from svsutils import cpramdisk
from svsutils import Slide
from svsutils import reinhard
from svsutils import PythonIterator
from svsutils import TensorflowIterator

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import traceback
import time

from milk.eager import ClassifierEager
from milk.encoder_config import get_encoder_args

import tfmodels
import gleason_grade as gg

import argparse
import os

def write_times(timedst, successes, ntiles, total_time, fpss):
  print('Timing info --> {}'.format(timedst))
  with open(timedst, 'w+') as f:
    header = 'File\tTiles\tTotalTime\tFPS\n'
    f.write(header)
    for success, ntile, totalt, fps in zip(successes, ntiles, total_time, fpss):
      s = '{}\t{}\t{}\t{}\n'.format(success, ntile, totalt, fps)
      print(s)
      f.write(s)

def main(args, sess):
  # Define a compute_fn that should do three things:
  # 1. define an iterator over the slide's tiles
  # 2. compute an output with given model parameter
  # 3. asseble / gather the output
  #
  # compute_fn - function can define part of a computation
  # graph in eager mode -- possibly in graph mode.
  # We should completely reset the graph each call then
  # I still don't know how nodes are actually represented in memory
  # or if keeping them around has a real cost.

  def compute_fn(slide, args, sess=None):
    # assert tf.executing_eagerly()
    print('\n\nSlide with {}'.format(len(slide.tile_list)))

    # I'm not sure if spinning up new ops every time is bad.
    # In this example the iterator is separate from the 
    # infernce function, it can also be set up with the two
    # connected to skip the feed_dict
    tf_iterator = TensorflowIterator(slide, args).make_iterator()
    img_op, idx_op = tf_iterator.get_next()
    # prob_op = model(img_op)
    # sess.run(tf.global_variables_initializer())

    # The iterator can be used directly. Ququeing and multithreading
    # are handled in the backend by the tf.data.Dataset ops
    # for k, (img, idx) in enumerate(eager_iterator):
    k, nk = 0, 0
    while True:
      try:
        img, idx = sess.run([img_op, idx_op,])
        prob = model.inference(img)
        nk += img.shape[0]
        slide.place_batch(prob, idx, 'prob', mode='full', clobber=True)
        k += 1

        if k % 50 == 0:
          prstr = 'Batch #{:04d} idx:{} img:{} ({:2.2f}-{:2.2f}) prob:{} T {} \
          '.format(k, idx.shape, img.shape, img.min(), img.max(), prob.shape, nk)
          print(prstr)
          if args.verbose:
            print('More info: ')
            print('img: ', img.dtype, img.min(), img.max(), img.mean())
            pmax = np.argmax(prob, axis=-1).ravel()
            for u in range(args.n_classes):
              count_u = (pmax == u).sum()
              print('- class {:02d} : {}'.format(u, count_u))

      except tf.errors.OutOfRangeError:
        print('Finished.')
        print('Total: {}'.format(nk)) 
        break

      except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
        break

    # We've exited the loop. Clean up the iterator
    del tf_iterator, idx_op, img_op

    # slide.make_outputs()
    slide.make_outputs()
    ret = slide.output_imgs['prob']
    return ret

  # Set up the model first
  model = gg.get_model(args.model, sess, 
                       args.process_size, 
                       args.n_classes)
  # NOTE big time wasted because you have to initialize, 
  # THEN run the restore op to replace the already-created weights
  sess.run(tf.global_variables_initializer())
  model.restore(args.snapshot)

  # Read list of inputs
  with open(args.slides, 'r') as f:
    slides = [x.strip() for x in f]

  # Loop over slides; Record times
  nslides = len(slides)
  successes, ntiles, total_time, fpss = [], [], [], []
  for i, src in enumerate(slides):
    # Dirty substitution of the file extension give us the
    # destination. Do this first so we can just skip the slide
    # if this destination already exists.
    # Set the --suffix option to reflect the model / type of processed output
    dst = repext(src, args.suffix)
    if os.path.exists(dst):
      print('{} Exists.'.format(dst))
      continue

    # Loading data from ramdisk incurs a one-time copy cost
    rdsrc = cpramdisk(src, args.ramdisk)

    # Wrapped inside of a try-except-finally.
    # We want to make sure the slide gets cleaned from 
    # memory in case there's an error or stop signal in the 
    # middle of processing.
    try:
      # Initialze the side from our temporary path, with 
      # the arguments passed in from command-line.
      # This returns an svsutils.Slide object
      print('\n\n-------------------------------')
      print('File:', rdsrc, '{:04d} / {:04d}'.format(i, nslides))
      t0 = time.time()
      slide = Slide(rdsrc, args)

      # This step will eventually be included in slide creation
      # with some default compute_fn's provided by svsutils
      # For now, do it case-by-case, and use the compute_fn
      # that we defined just above.
      # TODO pull the expected output size from the model.. ? 
      # support common model types - keras, tfmodels, tfhub..
      slide.initialize_output('prob', args.n_classes, mode='full',
        compute_fn=compute_fn)

      # Call the compute function to compute this output.
      # Again, this may change to something like...
      #     slide.compute_all
      # which would loop over all the defined output types.
      ret = slide.compute('prob', args, sess=sess)
      print('{} --> {}'.format(ret.shape, dst))
      ret = (ret * 255).astype(np.uint8)
      np.save(dst, ret)

      # If it finishes, record some stats
      tend = time.time()
      deltat = tend - t0
      fps = len(slide.tile_list) / float(deltat)
      successes.append(rdsrc)
      ntiles.append(len(slide.tile_list))
      total_time.append(deltat)
      fpss.append(fps)
    except Exception as e:
      print(e)
      traceback.print_tb(e.__traceback__)
    finally:
      print('Removing {}'.format(rdsrc))
      os.remove(rdsrc)
      try:
        print('Cleaning slide object')
        slide.close()
        del slide
      except:
        print('No slide object not found to clean up ?')

  write_times(args.timefile, successes, ntiles, total_time, fpss)

if __name__ == '__main__':
  """
  standard __name__ == __main__ ?? 
  how to make this nicer
  """
  p = argparse.ArgumentParser()
  # positional arguments for this program
  # last call
  # python Example_classifier.py
  
  p.add_argument('slides') 
  p.add_argument('model') # [densenet/_s, fcn8s, fcn8s/_s, unet/_s]
  p.add_argument('snapshot') 
  p.add_argument('--iter_type', default='py', type=str) 
  p.add_argument('--suffix', default='.prob.npy', type=str) 
  p.add_argument('--timefile', default='times.txt', type=str) 

  # common arguments with defaults
  p.add_argument('-b', dest='batchsize', default=64, type=int)
  p.add_argument('-r', dest='ramdisk', default='/dev/shm', type=str)
  p.add_argument('-j', dest='workers', default=6, type=int)
  p.add_argument('-c', dest='n_classes', default=4, type=int)

  # Slide options
  p.add_argument('--mag',   dest='process_mag', default=5, type=int)
  p.add_argument('--chunk', dest='process_size', default=96, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.25, type=float)
  p.add_argument('--verbose', dest='verbose', default=False, action='store_true')

  args = p.parse_args()

  # Functionals for later:
  args.__dict__['preprocess_fn'] = lambda x: (x * (2/255.)-1).astype(np.float32)

  tfconfig = tf.ConfigProto()
  tfconfig.gpu_options.allow_growth = True
  with tf.Session(config=tfconfig) as sess:
    main(args, sess)
