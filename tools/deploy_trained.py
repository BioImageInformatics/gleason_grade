from __future__ import print_function
import os
import cv2
import sys
import glob
import time
import shutil
import argparse

import numpy as np
import tensorflow as tf

# sys.path.insert(0, 'svs_reader')
from svs_reader import Slide
# sys.path.insert(0, 'tfmodels')
import tfmodels

sys.path.insert(0, '.')

from densenet import Inference as densenet
from densenet_small import Inference as densenet_s
from fcn8s import Inference as fcn8s
from fcn8s_small import Inference as fcn8s_s
from unet import Inference as unet
from unet_small import Inference as unet_s

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

PRINT_ITER = 500
RAM_DISK = '/dev/shm'

def preprocess_fn(img):
  img = img * (2/255.) -1
  return img.astype(np.float32)

def prob_output(svs):
  probs = svs.output_imgs['prob']
  ## quantize to [0, 255] uint8
  probs *= 255.
  probs = probs.astype(np.uint8)
  return probs
  # return probs[:,:,:3]

def rgb_output(svs):
  rgb = svs.output_imgs['rgb']
  rgb += 1.0
  rgb *= (255. / 2.)
  # print 'fixed: ', rgb.shape, rgb.dtype, rgb.min(), rgb.max()
  return rgb[:,:,::-1]

def transfer_to_ramdisk(src, ramdisk = RAM_DISK):
  base = os.path.basename(src)
  dst = os.path.join(ramdisk, base)
  shutil.copyfile(src, dst)
  return dst

def process_slide(slide_path, model, sess, out_dir, process_mag, 
          process_size, oversample, batch_size, n_classes):
  """ Process a slide

  Args:
  slide_path: str
    absolute or relative path to svs formatted slide
  model: tfmodels.SegmentationBasemodel object
    model definition to use. Weights must be restored first, 
    i.e. call model.restore() before passing model
  sess: tf.Session
  out_dir: str
    path to use for output
  process_mag: int
    Usually one of: 5, 10, 20, 40.
    Other values may work but have not been tested
  process_size: int
    The input size required by model. 
  oversample: float. Usually in [1., 2.]
    How much to oversample between tiles. Larger values 
    will increase processing time.
  batch_size: int
    The batch size for inference. If the batch size is too 
    large given the model and process_size, then OOM errors
    will be raised
  n_classes: int
    The number of classes output by model. 
    i.e. shape(model.yhat) = (batch, h, w, n_classes)
  """
  
  print('Working {}'.format(slide_path))
  svs = Slide(slide_path  = slide_path,
        preprocess_fn = preprocess_fn,
        process_mag   = process_mag,
        process_size  = process_size,
        oversample  = oversample,
        verbose = False,
        )
  svs.initialize_output('prob', dim=n_classes)
  svs.initialize_output('rgb', dim=3)
  PREFETCH = min(len(svs.place_list), 1024)

  def wrapped_fn(idx):
    try:
      coords = svs.tile_list[idx]
      img = svs._read_tile(coords)
      return img, idx
    except:
      return 0

  def read_region_at_index(idx):
    return tf.py_func(func = wrapped_fn,
              inp  = [idx],
              Tout = [tf.float32, tf.int64],
              stateful = False)

  ds = tf.data.Dataset.from_generator(generator=svs.generate_index,
    output_types=tf.int64)
  ds = ds.map(read_region_at_index, num_parallel_calls=12)
  ds = ds.prefetch(PREFETCH)
  ds = ds.batch(batch_size)

  iterator = ds.make_one_shot_iterator()
  img, idx = iterator.get_next()

  print('Processing {} tiles'.format(len(svs.tile_list)))
  tstart = time.time()
  n_processed = 0
  while True:
    try:
      tile, idx_ = sess.run([img, idx])
      output = model.inference(tile)
      svs.place_batch(output, idx_, 'prob')
      svs.place_batch(tile, idx_, 'rgb')

      n_processed += BATCH_SIZE
      if n_processed % PRINT_ITER == 0:
        print('[{:06d}] elapsed time [{:3.3f}]'.format(
          n_processed, time.time() - tstart ))

    except tf.errors.OutOfRangeError:
      print('Finished')
      dt = time.time()-tstart
      spt = dt / float(len(svs.tile_list))
      fps = len(svs.tile_list) / dt
      print('\nFinished. {:2.2f}min {:3.3f}s/tile\n'.format(dt/60., spt))
      print('\t {:3.3f} fps\n'.format(fps))

      svs.make_outputs()
      prob_img = prob_output(svs)
      rgb_img = rgb_output(svs)
      break

    except Exception as e:
      print('Caught exception at tiles {}'.format(idx_))
      # print(e.__doc__)
      # print(e.message)
      prob_img = None
      rgb_img = None
      break

  svs.close()

  return prob_img, rgb_img, fps


def _get_model(model_type, sess, process_size):
  """ Return a model instance to use 

  """
  x_dims = [process_size, process_size, 3]
  if model_type == 'densenet':
    model = densenet(sess=sess, x_dims=x_dims)
  if model_type == 'densenet_s':
    model = densenet_s(sess=sess, x_dims=x_dims)
  if model_type == 'fcn8s':
    model = fcn8s(sess=sess, x_dims=x_dims)
  if model_type == 'fcn8s_s':
    model = fcn8s_s(sess=sess, x_dims=x_dims)
  if model_type == 'unet':
    model = unet(sess=sess, x_dims=x_dims)
  if model_type == 'unet_s':
    model = unet_s(sess=sess, x_dims=x_dims)

  return model


def read_list(slide_list):
  with open(slide_list, 'r') as f
    lst = [x.strip() for x in f]
  return lst

def main(args):
  assert args.out is not None
  # assert args.slide_dir is not None
  out_dir = args.out

  # slide_list = glob.glob(os.path.join(args.slide_dir, '*svs'))
  slide_list = read_list(args.slide_list)
  print('Working on {} slides from {}'.format(len(slide_list), args.slide_dir))

  if not os.path.exists(out_dir):
    print('Creating {}'.format(out_dir))
    os.makedirs(out_dir)

  processed_list = glob.glob(os.path.join(out_dir, '*_prob.npy'))
  print('Found {} processed slides.'.format(len(processed_list)))
  processed_base = [os.path.basename(x).replace('_prob.npy', '') for x in processed_list]

  slide_base = [os.path.basename(x).replace('.svs', '') for x in slide_list]
  slide_base_list = zip(slide_base, slide_list)
  slide_list = [lst for bas, lst in slide_base_list if bas not in processed_base]
  print('Trimmed processed slides. Working on {}'.format(len(slide_list)))

  print('out_dir: ', out_dir)
  with tf.Session(config=config) as sess:
    model = _get_model(args.model, sess, args.size)
    try:
      model.restore(args.snapshot)
    except:
      raise Exception('Snapshot restore failed. model={} snapshot={}'.format(
        args.model, args.snapshot
      ))

    times = {}
    fpss = {}
    for slide_num, slide_path in enumerate(slide_list):
      print('\n\n[\tSlide {}/{}\t]\n'.format(slide_num, len(slide_list)))
      ramdisk_path = transfer_to_ramdisk(slide_path)
      try:
        time_start = time.time()
        prob_img, rgb_img, fps =  process_slide(ramdisk_path, model, sess, out_dir,
          args.mag, args.size, args.oversample, args.batch_size, args.n_classes)
        if prob_img is None:
          raise Exception('Failed.')

        outname_prob = os.path.basename(ramdisk_path).replace('.svs', '_prob.npy')
        outname_rgb = os.path.basename(ramdisk_path).replace('.svs', '_rgb.jpg')

        outpath =  os.path.join(out_dir, outname_prob)
        print('Writing {}'.format(outpath))
        np.save(outpath, prob_img)

        outpath =  os.path.join(out_dir, outname_rgb)
        print('Writing {}'.format(outpath))
        cv2.imwrite(outpath, rgb_img)
        times[ramdisk_path] = (time.time() - time_start) / 60.
        fpss[ramdisk_path] = fps

      except Exception as e:
        print('Caught exception')
        print(e.__doc__)
        print(e.message)
      finally:
        os.remove(ramdisk_path)
        print('Removed {}'.format(ramdisk_path))

  time_record = os.path.join(out_dir, 'processing_time.txt')
  fps_record = os.path.join(out_dir, 'processing_fps.txt')
  print('Writing processing times to {}'.format(time_record))
  times_all = []
  with open(time_record, 'w+') as f:
    for slide, tt in times.items():
      times_all.append(tt)
      f.write('{}\t{:3.5f}\n'.format(slide, tt))

    times_mean = np.mean(times_all)
    times_std = np.std(times_all)
    f.write('Mean: {:3.4f} +/- {:3.5f}\n'.format(times_mean, times_std))

  fps_all = []
  print('Writing processing FPS to {}'.format(fps_record))
  with open(fps_record, 'w+') as f:
    for slide, tt in fpss.items():
      fps_all.append(tt)
      f.write('{}\t{:3.5f}\n'.format(slide, tt))

    fps_mean = np.mean(fps_all)
    fps_std = np.std(fps_all)
    f.write('Mean: {:3.4f} +/- {:3.5f}\n'.format(fps_mean, fps_std))

  print('Done!')

if __name__ == '__main__':
  # Defaults
  PROCESS_MAG = 10
  PROCESS_SIZE = 256
  OVERSAMPLE = 1.2
  BATCH_SIZE = 8
  N_CLASSES = 5

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='densenet')
  parser.add_argument('--snapshot', default='./densenet.ckpt-61690')
  parser.add_argument('--slide_list', default=None)
  parser.add_argument('--out', default=None)

  parser.add_argument('--mag', default=PROCESS_MAG, type=int)
  parser.add_argument('--size', default=PROCESS_SIZE, type=int)
  parser.add_argument('--n_classes', default=N_CLASSES, type=int)
  parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
  parser.add_argument('--oversample', default=OVERSAMPLE, type=float)

  args = parser.parse_args()
  main(args)