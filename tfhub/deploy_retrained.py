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

sys.path.insert(0, '..')
from svs_reader import Slide

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def preprocess_fn(img):
    img = img * (1/255.)
    return img.astype(np.float32)

def prob_output(svs):
    probs = svs.output_imgs['prob']
    return probs

def transfer_to_ramdisk(src, ramdisk = '/dev/shm'):
    base = os.path.basename(src)
    dst = os.path.join(ramdisk, base)
    shutil.copyfile(src, dst)
    return dst

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

PROCESS_MAG = 10
PREFETCH = 2048
BATCH_SIZE = 4
OVERSAMPLE = 1.25
PRINT_ITER = 500
def main(sess, slide_path, image_op, predict_op):
    input_size = image_op.get_shape().as_list()
    print(input_size)
    x_size, y_size = input_size[1:3]

    print('Working {}'.format(slide_path))
    svs = Slide(slide_path    = ramdisk_path,
                preprocess_fn = preprocess_fn,
                process_mag   = PROCESS_MAG,
                process_size  = x_size,
                oversample_factor = OVERSAMPLE)
    svs.initialize_output('prob', dim=5, mode='tile')
    svs.print_info()
    PREFETCH = min(len(svs.tile_list), 2048)

    def wrapped_fn(idx):
        coords = svs.tile_list[idx]
        img = svs._read_tile(coords)
        return img, idx

    def read_region_at_index(idx):
        return tf.py_func(func = wrapped_fn,
                          inp  = [idx],
                          Tout = [tf.float32, tf.int64],
                          stateful = False)

    ds = tf.data.Dataset.from_generator(generator=svs.generate_index,
        output_types=tf.int64)
    ds = ds.map(read_region_at_index, num_parallel_calls=8)
    ds = ds.prefetch(PREFETCH)
    ds = ds.batch(BATCH_SIZE)

    iterator = ds.make_one_shot_iterator()
    img, idx = iterator.get_next()

    print('Processing {} tiles'.format(len(svs.tile_list)))
    tstart = time.time()
    n_processed = 0
    while True:
        try:
            tile, idx_ = sess.run([img, idx])
            output = sess.run(predict_op, {image_op: tile})
            svs.place_batch(output, idx_, 'prob', mode='tile')

            n_processed += BATCH_SIZE
            if n_processed % PRINT_ITER == 0:
                print('[{:06d}] elapsed time [{:3.3f}] ({})'.format(
                    n_processed, time.time() - tstart, tile.shape ))

        except tf.errors.OutOfRangeError:
            print('Finished')
            break

    dt = time.time()-tstart
    spt = dt / float(len(svs.tile_list))
    print('\nFinished. {:2.2f}min {:3.3f}s/tile\n'.format(dt/60., spt))
    print('\t {:3.3f} fps\n'.format(len(svs.tile_list) / dt))

    prob_img = prob_output(svs)
    svs.close()

    return prob_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--slide_dir')
    parser.add_argument('--out')

    args = parser.parse_args()
    model_path = args.model_path
    slide_dir = args.slide_dir
    out_dir = args.out
    print(args)

    print(slide_dir)
    slide_list = glob.glob(os.path.join(slide_dir, '*.svs'))
    print('Slide list: {}'.format(len(slide_list)))

    if not os.path.exists(out_dir):
        print('Making {}'.format(out_dir))
        os.makedirs(out_dir)

    with tf.Session(config=config) as sess:

        image_op , predict_op = get_input_output_ops(sess, model_path)
        for slide_path in slide_list:

            ramdisk_path = transfer_to_ramdisk(slide_path)
            try:
                prob_img = main(sess, ramdisk_path, image_op, predict_op)
                outname_prob = os.path.basename(ramdisk_path).replace('.svs', '_prob.npy')
                outpath =  os.path.join(out_dir, outname_prob)
                print('Writing {}'.format(outpath))
                np.save(outpath, prob_img)

            except Exception as e:
                print('Caught exception')
                print(e.__doc__)
                print(e.message)

            finally:
                os.remove(ramdisk_path)
                print('Removed {}'.format(ramdisk_path))
