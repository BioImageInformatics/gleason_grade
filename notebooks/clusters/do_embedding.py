#!/usr/bin/env python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import shutil
import glob
import cv2
import os

import MulticoreTSNE as mtsne
import umap

ZPATH = 'nasnet-large/z.npy'
EPATH = 'nasnet-large/tsne.npy'

data = np.load(ZPATH)
print(ZPATH, '-->', data.shape)

embedder = mtsne.MulticoreTSNE(n_jobs = 8,
    n_iter=2000,
    perplexity=10)
# embedder = umap.UMAP()
emb = embedder.fit_transform(data)
print(emb.shape, '-->', EPATH)
np.save(EPATH, emb)