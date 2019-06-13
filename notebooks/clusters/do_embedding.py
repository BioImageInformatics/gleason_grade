#!/usr/bin/env python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import shutil
import glob
import cv2
import os

import MulticoreTSNE as mtsne

ZPATH = 'tsne/inception_v3/z.npy'
EPATH = 'tsne/inception_v3/e.npy'

embedder = mtsne.MulticoreTSNE(n_jobs = 8)
data = np.load(ZPATH)

print(ZPATH, '-->', data.shape)

emb = embedder.fit_transform(data)

print(emb.shape, '-->', EPATH)
np.save(EPATH, emb)