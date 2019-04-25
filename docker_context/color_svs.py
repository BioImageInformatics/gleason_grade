from __future__ import print_function
import os
import cv2
import glob
import argparse
import numpy as np

colors = np.array([[175, 33, 8],
                   [20, 145, 4],
                   [177, 11, 237],
                   [14, 187, 235],
                   [3, 102, 163],
                   [0,0,0]
                  ])

def main(args):
  serch = os.path.join(args.srcdir, '*.npy')
  inf_list = glob.glob(serch)  # Trained models
  for n, inf_path in enumerate(sorted(inf_list)):
      inf_out = inf_path.replace('prob.npy', 'color.jpg')
      if os.path.exists(inf_out):
          print('{} {} Exists'.format(n, inf_out))
          continue
          
      x = np.load(inf_path)

      mask = np.zeros(list(x.shape[:2])+[3], dtype=np.uint8) 
      xsum = np.sum(x, axis=-1)
      amax = np.argmax(x, axis=-1)
      amax[xsum < 1.-1e-3] = 5

      for k in range(5):
          mask[amax==k] = colors[k,:]

      print(n, inf_out)
      cv2.imwrite(inf_out, mask[:,:,::-1])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--srcdir')

  args = parser.parse_args()
  main(args)
