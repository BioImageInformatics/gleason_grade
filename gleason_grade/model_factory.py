from .densenet import Inference as densenet
from .densenet_small import Inference as densenet_s
from .fcn8s import Inference as fcn8s
from .fcn8s_small import Inference as fcn8s_s
from .unet import Inference as unet
from .unet_small import Inference as unet_s

from .densenet import Training as densenet_t
from .densenet_small import Training as densenet_s_t
from .fcn8s import Training as fcn8s_t
from .fcn8s_small import Training as fcn8s_s_t
from .unet import Training as unet_t
from .unet_small import Training as unet_s_t

def get_model(model_type, sess, process_size, 
              n_classes, training=False):
  """ Return a model instance to use 

  This function can use a workover to be smarter

  """
  x_dims = [process_size, process_size, 3]
  argin = {'sess': sess, 
          'x_dims': x_dims, 
          'n_classes': n_classes}
  if training:
    # If training, return a class
    if model_type == 'densenet':
      model = densenet_t
    if model_type == 'densenet_s':
      model = densenet_s_t
    if model_type == 'fcn8s':
      model = fcn8s_t
    if model_type == 'fcn8s_s':
      model = fcn8s_s_t
    if model_type == 'unet':
      model = unet_t
    if model_type == 'unet_s':
      model = unet_s_t

  else:

    if model_type == 'densenet':
      model = densenet(**argin)
    if model_type == 'densenet_s':
      model = densenet_s(**argin)
    if model_type == 'fcn8s':
      model = fcn8s(**argin)
    if model_type == 'fcn8s_s':
      model = fcn8s_s(**argin)
    if model_type == 'unet':
      model = unet(**argin)
    if model_type == 'unet_s':
      model = unet_s(**argin)

  return model

def link_snapshot(argstr):
  """ Parse the arguments 
  return candidate snapshots from the model zoo
  """
  pass
