---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from __future__ import print_function
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import sys
import os
import glob

%matplotlib inline
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans)
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, classification_report, f1_score)

from MulticoreTSNE import MulticoreTSNE as MTSNE
import umap

from svs_reader import reinhard
```

```python
## These are the pretrained modules
MODULE_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/1'; MODULE_NAME='MobilenetV2'

# config = tf.ConfigProto( device_count = {'GPU': 0} )
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

module = hub.Module(MODULE_URL)
height, width = hub.get_expected_image_size(module)
print(height, width)
image_in = tf.placeholder('float', [1, height, width, 3])
z_op = module(image_in)
sess.run(tf.global_variables_initializer())
```

```python
DATA_HOME = '../../data'
jpg_list =  sorted(glob.glob(os.path.join(DATA_HOME, 'val_jpg_ext/*.jpg')))
mask_list = sorted(glob.glob(os.path.join(DATA_HOME, 'val_mask_ext/*.png')))

z_vectors = []
idx = 0

# Images at 20X; resize by 1/2 to get 10X
resize = 0.5
crop_size = int(height * (1/resize))

samples = 7
x0_vect = np.linspace(0, 1200-crop_size, samples, dtype=np.int)
y0_vect = np.linspace(0, 1200-crop_size, samples, dtype=np.int)
coords = zip(x0_vect, y0_vect)

for img_idx, (jpg, mask) in enumerate(zip(jpg_list, mask_list)): 
    y = cv2.imread(mask, -1)
    x = cv2.imread(jpg, -1)[:,:,::-1]
    x = reinhard(x)
               
    for k in range(samples):
        x0 = x0_vect[k]
        y0 = y0_vect[k]
        
        ## Our annotations are going to be the majority label
        ## when we tally up the labels in a mask
        ## Grab the majority label
        y_ = y[x0:x0+crop_size, y0:y0+crop_size]
        totals = np.zeros(5)
        for k in range(5):
            totals[k] = (y_==k).sum()

        # Check for majority
        maj = np.argmax(totals)   
        if totals[maj] > 0.5 * (crop_size**2):
            # check for stroma and skip
            if maj==4 and totals[maj] < 0.95 * (crop_size*2):
                continue
        else:
            # if there's no consensus (> 50% label) in this piece of the image, skip it.
            continue

        idx += 1
        if idx % 500 == 0:
            print('{} [{} / {}]'.format(idx, img_idx, len(jpg_list)))
        x_ = x[x0:x0+crop_size, y0:y0+crop_size, :]
        x_ = cv2.resize(x_, dsize=(0,0), fx=resize, fy=resize)
        x_ = x_ * (1/255.)
        x_ = np.expand_dims(x_, 0)
        
        z_ = sess.run(z_op, feed_dict={image_in: x_.astype(np.float32)})
        z_vectors.append(z_)
        
z_vectors = np.concatenate(z_vectors, axis=0)
print('z vectors', z_vectors.shape)
z_manifold = umap.UMAP().fit_transform(z_vectors)
print('z manifold', z_manifold.shape)

np.save('mobilenet_z_manifold.npy', z_manifold)
np.save('mobilenet_z_vectors.npy', z_vectors)

print('Done')
```

```python

```
