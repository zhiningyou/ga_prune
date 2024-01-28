import tensorflow as tf

from .registry import register
from .defaults import default, default_cifar10


@register
def mobilenetv2_default():
    hps = default_cifar10()
    hps.initializer = "glorot_uniform_initializer"
    hps.model = "mobilenetv2"
    hps.learning_rate = 0.001
    hps.lr_scheme = "constant"
    hps.weight_decay_rate = 0.0005
    hps.num_classes = 10
    hps.optimizer = "adam"
    hps.adam_epsilon = 1e-6
    hps.beta1 = 0.85
    hps.beta2 = 0.997
    hps.input_shape = [32, 32, 3]
    hps.output_shape = [10]
    hps.batch_size = 20
    hps.dropout_type = "prune_with_mask"
    return hps


@register
def mobilenetv2_mask():
    hps = mobilenetv2_default()
    hps.data = "cifar10"

    hps.input_shape = [32, 32, 3]
    hps.output_shape = [100]
    hps.num_classes = 100
    hps.channels = 3
    hps.learning_rate = 0.001
    hps.dropout_type = "prune_with_mask"
    return hps
