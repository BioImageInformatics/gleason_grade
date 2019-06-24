#!/usr/bin/env python

"""

"""

import argparse
import cv2
import glob
import os

import numpy as np
# colors = np.array([[255, 255, 57], # Bright yellow
#                    [198, 27, 27],  # Red
#                    [11, 147, 8],   # Green
#                    [252, 141, 204],# Pink
#                    [255, 255, 255],
#                   ])
colors = np.array([
    (130,84,45),   # brown
    (214,7,36),    # red
    (37,131,135),   # turquois
    (244,202,203), # pink
    (255,255,255)
])
mixture = [0.3, 0.7]

# TODO fix to always return sorted lists
def matching_basenames(l1, l2):
    l1_base = [os.path.splitext(os.path.basename(x))[0].split('_')[0] for x in l1]
    l2_base = [os.path.splitext(os.path.basename(x))[0].split('_')[0] for x in l2]
    matching = np.intersect1d(l1_base, l2_base)
    l1_out = []
    for l in l1:
        b = os.path.splitext(os.path.basename(l))[0].split('_')[0]
        if b in matching: l1_out.append(l)
    l2_out = []
    for l in l2:
        b = os.path.splitext(os.path.basename(l))[0].split('_')[0]
        if b in matching: l2_out.append(l)
    return l1_out, l2_out


def color_mask(mask):
    uq = np.unique(mask)
    r = np.zeros(shape=mask.shape, dtype=np.uint8)
    g = np.copy(r)
    b = np.copy(r)
    for u in uq:
        r[mask==u] = colors[u,0]
        g[mask==u] = colors[u,1]
        b[mask==u] = colors[u,2]
    newmask = np.dstack((b,g,r))
    return newmask


def overlay_img(base, pred):
    img = cv2.imread(base)
    ishape = img.shape[:2][::-1]
    y = np.load(pred)
    y = cv2.resize(y, fx=0, fy=0, dsize=ishape, interpolation=cv2.INTER_LINEAR)
    ymax = np.argmax(y, axis=-1)

    # Find unprocessed space
    ymax[np.sum(y, axis=-1) < 1e-2] = 4 # white

    # Find pure black and white in the img
    gray = np.mean(img, axis=-1)
    img_w = gray > 220
    img_b = gray < 10

    ymax = color_mask(ymax)
    img = np.add(img*mixture[0], ymax*mixture[1])
    channels = np.split(img, 3, axis=-1)
    for c in channels:
        c[img_w] = 255
        c[img_b] = 255
    img = np.dstack(channels)
    return cv2.convertScaleAbs(img)


def main(args):
    baseimgs = sorted(glob.glob('{}/*rgb.jpg'.format(args.s)))
    predictions = sorted(glob.glob('{}/*prob.npy'.format(args.p)))
    baseimgs, predictions = matching_basenames(baseimgs, predictions)

    for bi, pr in zip(baseimgs, predictions):
        combo = overlay_img(bi, pr)
        dst = pr.replace('prob.npy', 'overlay.jpg')
        print('{} --> {}'.format(combo.shape, dst))
        cv2.imwrite(dst, combo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', default='hires_imgs', type=str)
    parser.add_argument('-p', default='sbu_inference', type=str)
    # parser.add_argument('-d', default='hires_imgs', type=str)

    args = parser.parse_args()
    main(args)
