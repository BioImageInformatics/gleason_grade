from svsutils import define_colors, overlay_img, repext
import argparse
import sys
import cv2
import os

def main(args):
  colors = define_colors(args.colorname, 
                         args.n_colors,
                         add_white = True, 
                         shuffle = False)
  # print(colors)

  dst = f'{args.prob}.ovr.jpg'
  if os.path.exists(dst):
    print('{} Exists.'.format(dst))
    sys.exit(1)

  print(args.slide, args.prob, '-->', dst)
  ret = overlay_img(args.slide, args.prob, colors, 
        mixture = [0.3, 0.7])
  cv2.imwrite(dst, ret)

if __name__ == '__main__':
  """
  standard __name__ == __main__ ?? 
  how to make this nicer
  """
  p = argparse.ArgumentParser()
  # positional arguments for this 
  p.add_argument('slide', help = 'Path to whole slide image (svs)') 
  p.add_argument('prob' , help = 'Path to probability mask (npy)') 

  # optional long name arguments
  p.add_argument('--colorname', default='elf', type=str)
  p.add_argument('--n_colors',  default=4, type=int)

  p.add_argument('-b', dest='batch', default=1, type=int)

  # Slide options
  p.add_argument('--mag',   dest='process_mag', default=5, type=int)
  p.add_argument('--chunk', dest='process_size', default=512, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.5, type=float)

  args = p.parse_args()
  main(args)