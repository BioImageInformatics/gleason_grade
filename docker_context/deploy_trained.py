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
from svs_reader import Slide, reinhard

sys.path.insert(0, '.')
import tfmodels

from densenet import Inference as densenet
# from densenet_small import Inference as densenet_s
# from fcn8s import Inference as fcn8s
# from fcn8s_small import Inference as fcn8s_s
# from unet import Inference as unet
# from unet_small import Inference as unet_s

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

PRINT_ITER = 500
RAM_DISK = '/app'

def preprocess_fn(img):
  img = reinhard(img)
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

def process_slide(slide_path, fg_path, model, sess, out_dir, process_mag, 
          process_size, oversample, batch_size, n_classes):
  """ Process a slide

  Args:
  slide_path: str
    absolute or relative path to svs formatted slide
  fb_path: str
    absolute or relative path to foreground png
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
  print('Working {}'.format(fg_path))
  fgimg = cv2.imread(fg_path, 0)
  fgimg = cv2.morphologyEx(fgimg, cv2.MORPH_CLOSE, 
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
  svs = Slide(slide_path  = slide_path,
        background_speed = 'image', 
        background_image = fgimg,
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


def _get_model(model_type, sess, process_size, n_classes):
  """ Return a model instance to use 

  """
  x_dims = [process_size, process_size, 3]
  if model_type == 'densenet':
    model = densenet(sess=sess, x_dims=x_dims,
                     n_classes=n_classes)
  # if model_type == 'densenet_s':
  #   model = densenet_s(sess=sess, x_dims=x_dims)
  # if model_type == 'fcn8s':
  #   model = fcn8s(sess=sess, x_dims=x_dims)
  # if model_type == 'fcn8s_s':
  #   model = fcn8s_s(sess=sess, x_dims=x_dims)
  # if model_type == 'unet':
  #   model = unet(sess=sess, x_dims=x_dims)
  # if model_type == 'unet_s':
  #   model = unet_s(sess=sess, x_dims=x_dims)

  return model

def get_slide_from_list(slide_file):
  print(slide_file)
  slide_list = []
  with open(str(slide_file), 'r') as f:
    for L in f:
      L_ = L.replace('\n', '')
      slide_list.append(L_)
  return slide_list

def get_matching_fg(slide_list, fg_list):
  slide_base = np.array([os.path.basename(s).replace('.svs', '') for s in slide_list])
  fg_base = np.array([os.path.basename(s).replace('_fg.png', '') for s in fg_list])
  print(slide_base)
  print(fg_base)
  slide_, fg_ = [], []
  for i, s in enumerate(slide_base):
    if s in fg_base:
      idx = np.argwhere(fg_base == s)[0][0]
      print(s, idx)    
      slide_.append(slide_list[i])
      fg_.append(fg_list[idx])

  for s, f in zip(slide_, fg_):
    print(s, f)

  return slide_, fg_

def main(args):
  out_dir = args.out

  assert os.path.exists(args.slide_list), "{} does not exist".format(slide_list)
  # slide_list = glob.glob(os.path.join(args.slide_dir, '*svs'))
  slide_list = get_slide_from_list(args.slide_list)
  fg_list = get_slide_from_list(args.fg_list)
  slide_list, fg_list = get_matching_fg(slide_list, fg_list)
  print('Working on {} slides from {}'.format(len(slide_list), args.slide_list))

  print('out_dir: ', out_dir)
  if not os.path.exists(out_dir):
    print('Creating {}'.format(out_dir))
    os.makedirs(out_dir)

  # processed_list = glob.glob(os.path.join(out_dir, '*_prob.npy'))
  # print('Found {} processed slides.'.format(len(processed_list)))
  # processed_base = [os.path.basename(x).replace('_prob.npy', '') for x in processed_list]

  # slide_base = [os.path.basename(x).replace('.svs', '') for x in slide_list]
  # slide_base_list = zip(slide_base, slide_list)
  # slide_list = [lst for bas, lst in slide_base_list if bas not in processed_base]
  # print('Trimmed processed slides. Working on {}'.format(len(slide_list)))

  with tf.Session(config=config) as sess:
    model = _get_model(args.model, sess, args.size, args.n_classes)
    try:
      model.restore(args.snapshot)
    except:
      raise Exception('Snapshot restore failed. model={} snapshot={}'.format(
        args.model, args.snapshot
      ))

    times = {}
    fpss = {}
    for slide_num, (slide_path, fg_path) in enumerate(zip(slide_list, fg_list)):
      print('\n\n[\tSlide {}/{}\t]\n'.format(slide_num, len(slide_list)))
      slide_path_base = os.path.basename(slide_path)

      assert os.path.exists(fg_path), 'fg file not found'.format(fg_path)
      ramdisk_path = transfer_to_ramdisk(slide_path)
      print(ramdisk_path)
      print(os.listdir('/app'))
      try:
        time_start = time.time()
        prob_img, rgb_img, fps =  process_slide(ramdisk_path, fg_path, model, sess, out_dir,
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
        print('Finished with {}'.format(ramdisk_path))

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
  print(os.listdir('.'))
  print(__file__)

  # Defaults
  PROCESS_MAG = 10
  PROCESS_SIZE = 256
  OVERSAMPLE = 1.25
  BATCH_SIZE = 4
  N_CLASSES = 4

  parser = argparse.ArgumentParser()
  parser.add_argument('--slide_list', default='slide.txt', type=str)
  parser.add_argument('--fg_list', default='foreground.txt', type=str)
  parser.add_argument('--model', default='densenet')
  parser.add_argument('--out', default='/data/inference')
  parser.add_argument('--snapshot', default='./densenet.ckpt-360000')
  parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
  parser.add_argument('--mag', default=PROCESS_MAG, type=int)
  parser.add_argument('--size', default=PROCESS_SIZE, type=int)
  parser.add_argument('--oversample', default=OVERSAMPLE, type=float)
  parser.add_argument('--n_classes', default=N_CLASSES, type=int)

  args = parser.parse_args()
  main(args)
