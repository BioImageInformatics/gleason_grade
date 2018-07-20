from __future__ import print_function
from openslide import OpenSlide
import cv2
import numpy as np
import os
import glob
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--svs_dir')
parser.add_argument('--suffix', default='thumb')
args = parser.parse_args()

svs_dir = args.svs_dir
suffix = args.suffix

svs_list = sorted(glob.glob(os.path.join(svs_dir, '*.svs')))
print(len(svs_list))

for svs_path in svs_list:
    out_path = svs_path.replace('.svs', '_{}.jpg'.format(suffix))
    svs = OpenSlide(svs_path)
    label_image = np.asarray(svs.associated_images['label'])[:,:,:3]
    print(out_path)
    print(label_image.shape, label_image.dtype, label_image.min(), label_image.max())

    cv2.imwrite(out_path, label_image)
