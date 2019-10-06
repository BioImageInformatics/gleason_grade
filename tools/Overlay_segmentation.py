from svsutils import define_colors, overlay_img, repext
import argparse
import cv2
import os

def main(args):
  # these are loaded in order
  with open(args.slides, 'r') as f:
    slidelist =  [l.strip() for l in f]

  with open(args.probs, 'r') as f:
    problist =  [l.strip() for l in f]

  colors = define_colors(args.colorname, 
                         args.n_colors,
                         add_white = True, 
                         shuffle = False)
  print(colors)

  idx = 0
  for slide, prob in zip(slidelist, problist):
    dst = repext(prob, '.ovr.jpg')
    if os.path.exists(dst):
      print('{} Exists.'.format(dst))
      continue

    print(slide, '-->', dst)
    ret = overlay_img(slide, prob, colors, 
          mixture = [0.3, 0.7])
    cv2.imwrite(dst, ret)
    idx += 1
    if idx % 10 == 0:
      print(idx)

if __name__ == '__main__':
  """
  standard __name__ == __main__ ?? 
  how to make this nicer
  """
  p = argparse.ArgumentParser()
  # positional arguments for this 
  p.add_argument('slides') 
  p.add_argument('probs') 

  # optional long name arguments
  p.add_argument('--colorname', default='elf', type=str)
  p.add_argument('--n_colors',  default=4, type=int)

  p.add_argument('-b', dest='batchsize', default=1, type=int)
  p.add_argument('-r', dest='ramdisk', default='./', type=str)

  # Slide options
  p.add_argument('--mag',   dest='process_mag', default=5, type=int)
  p.add_argument('--chunk', dest='process_size', default=512, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.5, type=float)

  args = p.parse_args()
  main(args)