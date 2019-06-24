from .densenet import Training as DensenetTraining
from .densenet import Inference as DensenetInference

from .densenet_small import Training as DensenetSmallTraining
from .densenet_small import Inference as DensenetSmallInference

from .fcn8s import Training as FcnTraining
from .fcn8s import Inference as FcnInference

from .fcn8s_small import Training as FcnSmallTraining
from .fcn8s_small import Inference as FcnSmallInference

from .unet import Training as UnetTraining
from .unet import Inference as UnetInference

from .unet_small import Training as UnetSmallTraining
from .unet_small import Inference as UnetSmallInference

from .model_factory import get_model

__all__ = [
    # Utilities
    'get_model',

    # Models
    'DensenetTraining',
    'DensenetInference',
    'DensenetSmallTraining',
    'DensenetSmallInference',
    'FcnTraining',
    'FcnInference',
    'FcnSmallTraining',
    'FcnSmallInference',
    'UnetTraining',
    'UnetInference',
    'UnetSmallTraining',
    'UnetSmallInference'
]