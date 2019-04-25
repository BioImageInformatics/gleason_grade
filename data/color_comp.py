#!/usr/bin/python

import cv2, os, glob
import numpy as np

# colors = np.array([[245,30,30],
#                    [45,245,45],
#                    [170,40,180],
#                    [220,220,50],
#                    [255,255,255]])

colors = np.array([[175, 33, 8],
                   [20, 145, 4],
                #    [177, 11, 237],
                   [14, 187, 235],
                #    [3, 102, 163],
                   [255,255,255]
                  ])

def color_mask(mask):
    uq = np.unique(mask)
    newmask = np.zeros(shape = mask.shape, dtype = np.uint8)
    mask = mask[:,:,1]
    r = newmask[:,:,0]
    g = newmask[:,:,1]
    b = newmask[:,:,2]
    for u in uq:
       r[mask == u] = colors[u,0]
       g[mask == u] = colors[u,1]
       b[mask == u] = colors[u,2]

    #newmask = np.concatenate((r,g,b), axis = 1)
    newmask = np.dstack((b,g,r))
    return newmask

def colorize(img, mask):
    img = cv2.imread(img)
    mask = cv2.imread(mask)

    mask = color_mask(mask)

    img = np.add(img*0.6, mask*0.4)
    img = cv2.convertScaleAbs(img)
    return img


def main(imgsrc, msksrc, dst):
    imglist = sorted(glob.glob(os.path.join(imgsrc, '*.jpg')))
    msklist = sorted(glob.glob(os.path.join(msksrc, '*.png')))

    for img, msk in zip(imglist, msklist):
        print img, msk
        color = colorize(img, msk)
        img_base = os.path.basename(img)
        dstfile = os.path.join(dst, img_base)
        cv2.imwrite(dstfile, color)


if __name__ == '__main__':

    imgsrc = 'jpg_ext'
    msksrc = 'mask_ext'
    dst = 'colored_ext'

    main(imgsrc, msksrc, dst)
