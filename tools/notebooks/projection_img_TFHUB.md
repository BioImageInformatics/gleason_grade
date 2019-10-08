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

```python
# Work with the saved_model API to attain hooks for feeding the models
def get_input_output_ops(sess, model_path):
    input_key = 'image'
    output_key = 'prediction'
    print('Loading model {}'.format(model_path))
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    meta_graph_def = tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        model_path )
    signature = meta_graph_def.signature_def

    print('Getting tensor names:')
    image_tensor_name = signature[signature_key].inputs[input_key].name
    print('Input tensor: ', image_tensor_name)
    predict_tensor_name = signature[signature_key].outputs[output_key].name
    print('Output tensor:', predict_tensor_name)

    image_op = sess.graph.get_tensor_by_name(image_tensor_name)
    predict_op = sess.graph.get_tensor_by_name(predict_tensor_name)
    print('Input:', image_op.get_shape())
    print('Output:', predict_op.get_shape())
    return image_op, predict_op
```

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# path to saved_model.pb and variables from a trained model
MODULE_PATH = '../tfhub_mobilenet/all/snapshot-4class'
image_in, predict_op = get_input_output_ops(sess, MODULE_PATH)
_, height, width, _ = image_in.get_shape()
```

## Populate a matrix of feature vectors

And run the classifier on images

```python
DATA_HOME = '../../data'
## Need the sorted() since glob returns in a random order
jpg_list =  sorted(glob.glob(os.path.join(DATA_HOME, 'val_jpg_ext/*.jpg')))
mask_list = sorted(glob.glob(os.path.join(DATA_HOME, 'val_mask_ext/*.png')))

img_plotting = {}
img_classes = []
orig_imgs = []
y_vectors = []

idx = 0
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
        # Skip white
        gray = cv2.cvtColor(x_, cv2.COLOR_RGB2GRAY)
        if (gray > 220).sum() > 0.5*(crop_size**2):
            # More than half white
            continue
            
        x_ = x_ * (1/255.)
        x_ = np.expand_dims(x_, 0)
        
        # With low probability save an image for annotation
        if np.random.randn() < -1:
            img_plotting[idx] = x_
        
        ## Comment out to refresh the plotting images quickly
        yhat = sess.run(predict_op, feed_dict={image_in: x_.astype(np.float32)})
        y_vectors.append(yhat)
        
print('Done')
```

```python
# img_classes are 0-5 - adjust them down
# 2 --> combines class 1 and 2.
# 3 --> 2 and 4 --> 3 is a shift down.
img_classes = np.asarray(img_classes)
img_classes[img_classes==2] = 1
img_classes[img_classes==3] = 2
img_classes[img_classes==4] = 3

y_vectors = np.concatenate(y_vectors, axis=0)
```

```python
## Use a saved set of coordinates
# z_manifold = np.load('mobilenet_z_manifold.npy')

## Re-calculate the embedding
z_vectors = np.load('mobilenet_z_vectors.npy')
z_manifold = umap.UMAP(n_neighbors=20, min_dist=0.5).fit_transform(z_vectors)

print(z_manifold.shape)
```

```python
ymax = np.argmax(y_vectors, axis=-1)
for k in range(4):
    print((ymax==k).sum())
    
print(ymax.shape)
accuracy = np.mean(img_classes == ymax)
print(accuracy)
```

```python
fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=300)

# Draw the scatter plot
for k in range(4):
#     idx = img_classes==k   # real class
    idx = ymax==k # predicted class
    plt.scatter(z_manifold[idx,0], z_manifold[idx,1], c=COLOR_HEX[k], label=LABELS[k],
                alpha=0.7, s=3)
    
plt.xticks([])
plt.yticks([])

artists = []
indices = [k for k in img_plotting.keys()]

# Change the seed to change the plotted images
np.random.seed(1111)
np.random.shuffle(indices)
for k in indices[:10]:
    img_ = img_plotting[k]
    x,y = z_manifold[k]
    im = OffsetImage(np.squeeze(img_), zoom=0.125)
    ab = AnnotationBbox(im, (x,y), xycoords='data', pad=0.05, frameon=True,
                        bboxprops={'ec': COLOR_HEX[img_classes[k]], 'lw': 2})
    artists.append(ax.add_artist(ab))
    
# plt.savefig('MobileNet_imgs.png', bbox_inches='tight')
```

```python

```
