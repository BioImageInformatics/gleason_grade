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

import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
%matplotlib inline

# For permutation tests
from MulticoreTSNE import MulticoreTSNE as MTSNE
import umap

from svs_reader import reinhard

sys.path.insert(0, '../../gleason_grade/densenet_small')
from densenet_small import Inference

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

labels = ['G3', 'High grade', 'BN', 'ST']

SNAPSHOT = '../../gleason_grade/densenet_small/ext_4class_10x/snapshots/densenet.ckpt-62375'
height = width = 256
model = Inference(sess=sess, x_dims=(height, width, 3))
model.restore(SNAPSHOT)

image_in = model.x_in
bottleneck_op = model.intermediate_ops['05. Bottleneck']
yhat_op = model.y_hat
```

```python
COLORS = np.array(
    [(130,84,45),   # brown
     (214,7,36),    # red
     (37,131,135),   # turquois
     (244,202,203),  # pink
    ]
)
COLOR_HEX = [
    '#82552d',
    '#d60724',
    '#258487',
    '#f4cacb',
]

LABELS = ['G3', 'High Grade', 'BN', 'ST']
```

## Populate a matrix of feature vectors

```python
DATA_HOME = '../../data'
jpg_list =  sorted(glob.glob(os.path.join(DATA_HOME, 'val_jpg_ext/*.jpg')))
mask_list = sorted(glob.glob(os.path.join(DATA_HOME, 'val_mask_ext/*.png')))
print(len(jpg_list), len(mask_list))

img_plotting = {}
img_classes = []
orig_imgs = []

z_vectors = []
y_vectors = []
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

        img_classes.append(maj)
        orig_imgs.append(img_idx)
        
        idx += 1
        if idx % 500 == 0:
            print('{} [{} / {}]'.format(idx, img_idx, len(jpg_list)))
        x_ = x[x0:x0+crop_size, y0:y0+crop_size, :]
        x_ = cv2.resize(x_, dsize=(0,0), fx=resize, fy=resize)
        x_ = x_ * (2/255.) - 1
        x_ = np.expand_dims(x_, 0)
        
        if np.random.randn() < -3:
            img_plotting[idx] = (x_ + 1)/2.
        
        z, yhat = sess.run([bottleneck_op, yhat_op], feed_dict={image_in: x_, model.keep_prob: 1.})
        ymax = np.argmax(yhat, axis=-1)
        num_classes = np.zeros(5)
        for ck in range(5):
            num_classes[ck] = (ymax==ck).sum()
        
        y_vectors.append(num_classes)
        
        z_int = np.mean(z, axis=(1,2))
        z_vectors.append(z_int)
        
    
img_classes = np.asarray(img_classes)
orig_imgs = np.asarray(orig_imgs)

print('Done')
```

```python
# Get the majority predicted label in each image
ymax = []
for yk in y_vectors:
    perm = np.argsort(yk)
    ym = perm[-1]
    
    if ym == 4: # stroma
        ypct = yk[ym] / float((height*width))
        if ypct < 0.9:
            ym = perm[-2]
            
    ymax.append(ym)
    
ymax = np.array(ymax)
print(ymax)

for k in range(5):
    print(k, np.sum(img_classes==k), np.sum(ymax==k))
    
print()

## Shift labels down 1:
img_classes[img_classes == 2] = 1
img_classes[img_classes == 3] = 2
img_classes[img_classes == 4] = 3

ymax[ymax == 2] = 1
ymax[ymax == 3] = 2
ymax[ymax == 4] = 3

for k in range(5):
    print(k, np.sum(img_classes==k), np.sum(ymax==k))

accuracy = (ymax==img_classes).mean()
print(accuracy)
```

```python
# Load precomputed z embeddings / or load the vectors and re-compute the embeddings with new settings
z_vectors = np.load('densenet_z_vectors.npy')

# z_manifold = np.load('densenet_z_manifold.npy')
z_manifold = umap.UMAP(n_neighbors=15, min_dist=1).fit_transform(z_vectors)
```

```python
fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=300)
for k in range(4):
    idx = img_classes==k   # real class
#     idx = ymax==k # predicted class
    plt.scatter(z_manifold[idx,0], z_manifold[idx,1], color=COLOR_HEX[k], label=labels[k],
                alpha=0.7, s=3)
plt.xticks([])
plt.yticks([])
    
artists = []
indices = [x for x in img_plotting.keys()]

# Set the seed for reproducible image choices
np.random.seed(222)
np.random.shuffle(indices)
boxprops={'ec': 'r'}

for k in indices[:15]:
#     imgy = ymax[k]
    
    img_ = img_plotting[k]
    x,y = z_manifold[k]
    im = OffsetImage(np.squeeze(img_), zoom=0.125)
    ab = AnnotationBbox(im, (x,y), xycoords='data', pad=0.05, frameon=True, 
                        bboxprops={'ec': COLOR_HEX[img_classes[k]], 'lw': 2})
    artists.append(ax.add_artist(ab))
    
plt.savefig('DenseNet_imgs.png')
```
