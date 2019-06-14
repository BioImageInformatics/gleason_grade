#!/usr/bin/env python

import MulticoreTSNE as mtsne
import numpy as np
import umap
import os

import argparse

def filter_variance(z, var_thresh=0.3):
  variances = np.std(z, axis=0) ** 2
  passing = variances > var_thresh
  zout = z[:, passing]
  print('variance threshold: {} --> {}'.format(z.shape, zout.shape))
  return zout

def main(args):
  assert os.path.exists(args.source)

  if args.o is None:
    bn = os.path.dirname(args.source)
    dst = os.path.join(bn, '{}.npy'.format(args.method))
  else:
    assert not os.path.exists(args.o)
    dst = args.o
  print('Saving to {}'.format(dst))
  
  if args.method == 'tsne':
    embedder = mtsne.MulticoreTSNE(n_jobs=args.j, 
                                   perplexity=50, 
                                   learning_rate=250,
                                   verbose=args.verbose)
  elif args.method == 'umap':
    embedder = umap.UMAP()
  else:
    print('Defaulting method to tsne')
    embedder = mtsne.MulticoreTSNE(n_jobs=args.j)

  z = np.load(args.source)

  if args.variance is not None:
    z = filter_variance(z, args.variance)

  emb = embedder.fit_transform(z)
  print('{} --> {} --> {} --> {}'.format(args.source,
    z.shape, emb.shape, dst))

  np.save(dst, emb)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('source')
  parser.add_argument('-o', default=None, type=str)
  parser.add_argument('-j', default=12, type=int)
  parser.add_argument('--method', default='tsne', type=str)
  parser.add_argument('--variance', default=None, type=float)
  parser.add_argument('--verbose', default=False, action='store_true')

  args = parser.parse_args()
  main(args)