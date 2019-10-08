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
import sys
print(sys.executable)
```

```python
from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import sys
import os
import glob

# For permutation tests
from MulticoreTSNE import MulticoreTSNE as MTSNE
import umap

from svs_reader import reinhard

sys.path.insert(0, '../../gleason_grade/densenet_small')
from densenet_small import Inference

# config = tf.ConfigProto( device_count = {'GPU': 0} )
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

SNAPSHOT = '../../gleason_grade/densenet_small/ext_4class_10x/snapshots/densenet.ckpt-62375'
height = width = 256
model = Inference(sess=sess, x_dims=(height, width, 3))
model.restore(SNAPSHOT)

image_in = model.x_in
bottleneck_op = model.intermediate_ops['05. Bottleneck']
yhat_op = model.y_hat
```

## Populate a matrix of feature vectors

```python
DATA_HOME = '../../data'
jpg_list =  sorted(glob.glob(os.path.join(DATA_HOME, 'val_jpg_ext/*.jpg')))
mask_list = sorted(glob.glob(os.path.join(DATA_HOME, 'val_mask_ext/*.png')))
print(len(jpg_list), len(mask_list))

img_classes = []

z_vectors = []
idx = 0

resize = 0.5
crop_size = int(height * (1/resize))
print(crop_size)

samples = 7
x0_vect = np.linspace(0, 1200-crop_size, samples, dtype=np.int)
y0_vect = np.linspace(0, 1200-crop_size, samples, dtype=np.int)

x_y = [(x_, y_) for x_ in x0_vect for y_ in y0_vect]

coords = zip(x0_vect, y0_vect)

for img_idx, (jpg, mask) in enumerate(zip(jpg_list, mask_list)):
    y = cv2.imread(mask, -1)
    x = cv2.imread(jpg, -1)[:,:,::-1]
    x = reinhard(x)
               
    for x0,y0 in x_y:
        ## Grab the majority label
        y_ = y[x0:x0+crop_size, y0:y0+crop_size]
        totals = np.zeros(5)
        for k in range(5):
            totals[k] = (y_==k).sum()

        # Check for majority
        maj = np.argmax(totals)   
        if totals[maj] > 0.5 * (crop_size**2):
            # check for stroma -- two ways to skip stroma
            if maj==4 and totals[maj] < 0.95 * (crop_size**2):
                continue
        else:
            continue

        idx += 1
        if idx % 500 == 0:
            print('{} [{} / {}]'.format(idx, img_idx, len(jpg_list)))
        x_ = x[x0:x0+crop_size, y0:y0+crop_size, :]
        x_ = cv2.resize(x_, dsize=(0,0), fx=resize, fy=resize)
        x_ = x_ * (2/255.) - 1
        x_ = np.expand_dims(x_, 0)
        
        z, yhat = sess.run([bottleneck_op, yhat_op], feed_dict={image_in: x_, model.keep_prob: 1.})
        z_int = np.mean(z, axis=(1,2))
        z_vectors.append(z_int)
        
    
z_vectors = np.concatenate(z_vectors, axis=0)
print('z vectors', z_vectors.shape)

# manifold = MTSNE(n_jobs=8, n_components=2, verbose=1)
# z_manifold = manifold.fit_transform(z_vectors)
z_manifold = umap.UMAP(n_neighbors=20, min_dist=0.5).fit_transform(z_vectors)
print('z manifold', z_manifold.shape)

np.save('densenet_z_manifold.npy', z_manifold)
np.save('densenet_z_vectors.npy', z_vectors)

print('Done')
```

```python

```
