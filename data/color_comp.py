#!/usr/bin/env python

import cv2, os, glob
import numpy as np

# colors = np.array([[234, 228, 44], # Yellow
#                    [232, 144, 37],  # Orange
#                    [206, 29, 2],  # Red
#                    [161, 166, 168], # Gray
#                    [255, 255, 255],
#                   ])

colors = np.array(
    [(130,84,45),   # brown
     (214,7,36),    # red
    #  (15,236,36),    # green
     (37,131,135),   # turquois
     (244,202,203)]
   )

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

def colorize(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    print(img_path, mask_path, np.unique(mask), img.shape, mask.shape)
    if mask.shape[0] != img.shape[0]:
        tgt = img.shape[:2][::-1]
        mask = cv2.resize(mask, dsize=tgt, interpolation=cv2.INTER_NEAREST)
        print('Resized: {} {}'.format(img.shape, mask.shape))
    mask = color_mask(mask)

    img = np.add(img*0.4, mask*0.6)
    img = cv2.convertScaleAbs(img)
    return img


def main(imgsrc, msksrc, dst):
    # imglist = sorted(glob.glob(os.path.join(imgsrc, '*.jpg')))
    # msklist = sorted(glob.glob(os.path.join(msksrc, '*.png')))
    with open(imgsrc , 'r') as f:
        imglist = [x.strip() for x in f]
    with open(msksrc , 'r') as f:
        msklist = [x.strip() for x in f]

    print('imglist: {}'.format(len(imglist)))
    print('msklist: {}'.format(len(msklist)))

    for img, msk in zip(imglist, msklist):
        assert os.path.exists(img)
        assert os.path.exists(msk)

        img_base = os.path.basename(img)
        dstfile = os.path.join(dst, img_base)
        # if os.path.exists(dstfile):
        #     continue
        color = colorize(img, msk)
        cv2.imwrite(dstfile, color)

if __name__ == '__main__':

    imgsrc = 'img_list.txt'
    msksrc = 'mask_list.txt'
    dst = 'colored_imgs'

    main(imgsrc, msksrc, dst)
