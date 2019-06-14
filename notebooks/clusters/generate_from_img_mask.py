import cv2
import numpy as np

def generate_imgs(jpg_list, mask_list, samples=5, resize=0.5, crop_size=512):
  print('\nGenerating')
  print('samples', samples)
  print('resize', resize)
  print('crop_size', crop_size)
  print()

  samples = 5
  np.random.seed(999)
  x_samples = [np.random.randint(0, 600-crop_size) for _ in range(samples)]
  y_samples = [np.random.randint(0, 600-crop_size) for _ in range(samples)]

  idx = 0
  for jpg, mask in zip(jpg_list, mask_list):
    y = cv2.imread(mask, -1)
    x = cv2.imread(jpg, -1)[:,:,::-1]
    x = cv2.resize(x, dsize=(0,0), fx=resize, fy=resize)
    x = x * (1/255.)

    for s in range(samples):
    # for x0, y0 in coords:
      x0 = x_samples[s]
      y0 = y_samples[s]    

      ## Grab the majority label
      y_ = y[x0:x0+crop_size, y0:y0+crop_size]
      totals = np.zeros(5)
      for k in range(5):
        totals[k] = (y_==k).sum()

      maj = np.argmax(totals)   
      if totals[maj] > 0.5 * (crop_size**2):
        idx += 1
      else:
        continue

      # Grab the cropped image
      x_ = x[x0:x0+crop_size, y0:y0+crop_size, :]
      x_ = np.expand_dims(x_, 0)

      if idx % 250 == 0:
        print('index:', idx, 'coord:', x0, y0, 'img:', x_.shape, x_.dtype)

      yield x_, maj