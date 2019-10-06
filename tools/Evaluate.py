from __future__ import print_function
import cv2
import numpy as np
import pandas as pd
import os
import glob
import itertools
import argparse

from scipy.stats import mode
from sklearn.metrics import (f1_score, roc_curve, roc_auc_score)


BACKGROUND = 255
SKIP_CODES = [4,5,6,7, 255]
def get_regions(mask, thresh=500):
#   label_images = {x: np.zeros_like(mask) for x in range(5)}
  label_images = []
  label_codes = []
  
  # labels = np.unique(mask)
  image = (mask < BACKGROUND).astype(np.uint8)
  
  contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
  for cnt_num, _ in enumerate(contours):
    dummy_image = np.zeros_like(mask)
    cv2.drawContours(dummy_image, contours, cnt_num, 1, -1)
    ## Check area -- 
    pos_area = dummy_image.sum()
    if pos_area < thresh:
      continue
    dummy_image = dummy_image.astype(np.bool)
    cnt_label = mode(mask[dummy_image])[0][0]
    if cnt_label in SKIP_CODES:
      continue
    else:
      label_images.append(np.copy(dummy_image))
      label_codes.append(cnt_label)
  return label_images, label_codes

def load_mask(p):
  mask = cv2.imread(p, -1)
  # Reassign high grade=2 into high grade=1
  # print('Reassigning mask==2 to 1')
  mask[mask==2] = 1
  mask[mask==3] = 2
  mask[mask==4] = 3
  mask[mask > 3] = 255
  return mask

def load_prob(p, mask_x, mask_y, interp=cv2.INTER_LINEAR):
  x = np.load(p)
  x = cv2.resize(x, dsize=(mask_y, mask_x), interpolation=interp)
  return x

def get_base(p):
  b = os.path.basename(p).split('.')[0]
  return b
    

# grade_dict = {0: 'G3', 1: 'G4', 2: 'G5', 3: 'BN', 4: 'ST'}
# grade_dict = {0: 'G3', 1: 'HG',  2: 'BN', 3: 'ST'}
# attrib = ['Region', 'Slide', 'Region_area', 'TotalAcc', 'EpitheliumAcc', 
#       'EpitheliumF1', 'Class_Label', 'Stroma_Area']
# attrib += ['{}'.format(x) for y,x in grade_dict.items()]
def perform_comparison(inf_list, mask_list, args):
  grade_dict = {0: 'G3', 1: 'HG',  2: 'BN', 3: 'ST'}
  attrib = ['Region', 'Slide', 'Region_area', 'TotalAcc', 'EpitheliumAcc', 
        'EpitheliumF1', 'Class_Label', 'Stroma_Area']
  attrib += ['{}'.format(x) for y,x in grade_dict.items()]
  performance = {k: [] for k in attrib}
  
  print('Found matching output/annotations: ', len(inf_list))

  for idx, (inf_path, mask_path) in enumerate(zip(inf_list, mask_list)):
    inf_base = get_base(inf_path)
    mask = load_mask(mask_path)
    mask_x, mask_y = mask.shape[:2]
    x = load_prob(inf_path, mask_x, mask_y)
    label_images, label_codes = get_regions(mask)

    xmax = np.argmax(x, axis=-1)
    xmax[mask==255] = 255

    ## Loop over the present classes:
    region = 0
    for LI, LC in zip(label_images, label_codes):
      ## pull out annotated area
      xmax_region = xmax[LI]
      total_size = float(xmax_region.shape[0])

      ## pull out non-stroma
      stroma_area = (xmax_region == args.stroma_code).sum()
      xmax_region = xmax_region[xmax_region != args.stroma_code]
      non_stroma_size = float(xmax_region.shape[0])

      gt_region = np.zeros_like(xmax_region) + LC
      total_acc = (xmax_region == LC).sum() / total_size
      epithelium_acc = (xmax_region == LC).sum() / non_stroma_size
      epithelium_f1 = f1_score(gt_region, xmax_region, average='weighted')

      if args.verbose:
        print(
          '\t{}\t{}\t{}\t{}\tACC:{:3.3f}\tF1:{:3.3f}'.format(
            idx, region, grade_dict[LC], inf_base, epithelium_acc, epithelium_f1)
        )

      performance['Region'].append('{}_{:02d}'.format(inf_base, region))
      performance['Slide'].append(inf_base)
      performance['Region_area'].append(total_size)
      performance['TotalAcc'].append(total_acc)
      performance['EpitheliumAcc'].append(epithelium_acc)
      performance['EpitheliumF1'].append(epithelium_f1)
      performance['Class_Label'].append(grade_dict[LC])
      performance['Stroma_Area'].append(stroma_area)
      for co, gr in grade_dict.items():
        performance['{}'.format(gr)].append((xmax_region == co).sum())
      # print(idx, region, grade_dict[LC], inf_path, 'ACC:', epithelium_acc, 'F1:', epithelium_f1)
      region += 1

  perf_df = pd.DataFrame(performance, index=performance['Region'], columns=attrib[1:])
  if args.verbose:
    print(perf_df.head())

  print('PERF DF: {} --> {}'.format(perf_df.shape, args.outpath))
  perf_df.to_csv(args.outpath, sep=',')

  ## Averages
  summary_file = args.outpath + '.summary.txt'
  summary_cols = ['Region_area', 'TotalAcc', 'EpitheliumAcc', 'EpitheliumF1', 'Stroma_Area']
  avgs = [np.mean(perf_df[col]) for col in summary_cols]
  stds = [np.std(perf_df[col]) for col in summary_cols]
  print(avgs)
  print(stds)
  with open(summary_file, 'w+') as f:
    f.write('{}\t{}\t{}\t{}\t{}\n'.format(*summary_cols))
    f.write('{:3.4f}\t{:3.4f}\t{:3.4f}\t{:3.4f}\t{:3.4f}\n'.format(*avgs))
    f.write('{:3.4f}\t{:3.4f}\t{:3.4f}\t{:3.4f}\t{:3.4f}\n'.format(*stds))

def main(args):
  with open(args.probs, 'r') as f:
    problist = [line.strip() for line in f]

  with open(args.masks, 'r') as f:
    masklist = [line.strip() for line in f]

  print('{} probs'.format(len(problist)))
  print('{} masks'.format(len(masklist)))
  print('Destination: {}'.format(args.outpath))
  perform_comparison(problist, masklist, args)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('probs', type=str, help='An ordered list of probability npy nd-arrays.')
  parser.add_argument('masks', type=str, help='An ordered list of PNG label-images.')
  parser.add_argument('outpath', type=str, help='File name to save results.')

  parser.add_argument('--background', default=255, type=int, help='The index of background; not to be processed')
  parser.add_argument('--skip_codes', default=[4, 5, 6, 7, 255], nargs='+', type=int, 
    help='The index of background; not to be processed')
  parser.add_argument('--stroma_code', default=3, type=int, 
    help='The index of stoma; to be excluded.')
  parser.add_argument('--verbose', default=False, action='store_true')

  args = parser.parse_args()
  main(args)