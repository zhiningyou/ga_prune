import tensorflow as tf

from .registry import register
from .defaults import default, default_cifar10
 
  
@register
def vgg19_default():
  vgg_default = default_cifar10()
  vgg_default.initializer = "glorot_uniform_initializer"
  vgg_default.model = "vgg19"
  vgg_default.learning_rate = 0.01
  vgg_default.lr_scheme = "constant"
  vgg_default.weight_decay_rate = 0.0005
  vgg_default.num_classes = 10
  vgg_default.optimizer = "adam"
  vgg_default.adam_epsilon = 1e-6
  vgg_default.beta1 = 0.85
  vgg_default.beta2 = 0.997
  vgg_default.input_shape = [32, 32, 3]
  vgg_default.output_shape = [10]
  return vgg_default
  
@register
def cifar100_vgg19_mask():
  hps = vgg19_default()
  hps.data = "cifar100"
  
  hps.input_shape = [32, 32, 3]
  hps.output_shape = [100]
  hps.num_classes = 100
  hps.channels = 3
  hps.learning_rate = 0.0001
  hps.dropout_type="prune_with_mask"
  return hps  
