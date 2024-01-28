import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np

from ..utils import dropouts
from ..utils.activations import get_activation
from ..utils.dropouts import get_dropout, smallify_dropout
from ..utils.initializations import get_init
from ..registry import register
from ..utils import model_utils
from ..utils.model_utils import ModeKeys
from ...training import tpu


@register("mobilenetv2")
def get_mobilenetv2(hparams, lr):
  """Callable model function compatible with Experiment API.

          Args:
            params: a HParams object containing values for fields:
              use_bottleneck: bool to bottleneck the network
              num_residual_units: number of residual units
              num_classes: number of classes
              batch_size: batch size
              weight_decay_rate: weight decay rate
          """

  def mobilenetv2(features, labels, mode, params):
    if hparams.use_tpu and 'batch_size' in params.keys():
      hparams.batch_size = params['batch_size']

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    #is_training = False

    ilayer = 0

    def _l1():
      """L1 weight decay loss."""
      if hparams.l1_norm == 0:
        return 0

      costs = []
      for var in tf.trainable_variables():
        if "DW" in var.name and "logit" not in var.name:
          costs.append(tf.reduce_mean(tf.abs(var)))

      return tf.multiply(hparams.l1_norm, tf.add_n(costs))

    def _fully_connected(x, out_dim):
      """FullyConnected layer for final output."""
      prev_dim = np.product(x.get_shape().as_list()[1:])
      x = tf.reshape(x, [hparams.batch_size, prev_dim])
      w = tf.get_variable('DW', [prev_dim, out_dim])
      b = tf.get_variable(
          'biases', [out_dim], initializer=tf.zeros_initializer())
      return tf.nn.xw_plus_b(x, w, b)

    def separable_conv_block(x, output_channel_number, name):
        '''
        mobilenet 卷积块
        :param x:
        :param output_channel_number:  输出通道数量 output channel of 1*1 conv layer
        :param name:
        :return:
        '''
        with tf.variable_scope(name):
            # 获取图像通道数
            input_channel = x.get_shape().as_list()[-1]
            # 按照最后一维进行拆分 eg channel_wise_x: [channel1, channel2, ...]
            channel_wise_x = tf.split(x, input_channel, axis=3)
            output_channels = []
            # 针对 每一个 通道进行 卷积
            for i in range(len(channel_wise_x)):
                output_channel = model_utils.conv(
                    x=channel_wise_x[i],
                    filter_size=3,
                    out_filters=1,
                    strides=[1, 1, 1, 1],
                    hparams = hparams,
                    is_training=is_training,
                    name='conv_%d' % i)
                output_channel = tf.nn.relu(output_channel)
                # 将卷积后的通道保存到列表中
                output_channels.append(output_channel)
            # 合并整个列表
            concat_layer = tf.concat(output_channels, axis=3)
            # 再次进行一个 1 * 1 的卷积
            conv1_1 = model_utils.conv(
                x=concat_layer,
                filter_size=1,
                out_filters=output_channel_number,
                strides=[1, 1, 1, 1],
                hparams=hparams,
                is_training=is_training,
                name='conv1_1')
            conv1_1 = tf.nn.relu(conv1_1)

            return conv1_1

    x = features["inputs"]
    x = model_utils.batch_norm(x, hparams, is_training)
    conv1 = model_utils.conv(
        x=x,
        filter_size=3,
        out_filters=32,
        hparams=hparams,
        is_training=is_training,
        name='conv1')
    conv1 = tf.nn.relu(conv1)

    # 池化层 图像输出为: 16 * 16
    pooling1 = tf.layers.max_pooling2d(conv1,
                                       (2, 2),  # 核大小 变为原来的 1/2
                                       (2, 2),  # 步长
                                       name='pool1'
                                       )

    separable_2a = separable_conv_block(pooling1,
                                        32,
                                        name='separable_2a'
                                        )
    separable_2b = separable_conv_block(separable_2a,
                                        32,
                                        name='separable_2b'
                                        )

    pooling2 = tf.layers.max_pooling2d(separable_2b,
                                       (2, 2),  # 核大小 变为原来的 1/2
                                       (2, 2),  # 步长
                                       name='pool2'
                                       )

    separable_3a = separable_conv_block(pooling2,
                                        32,
                                        name='separable_3a'
                                        )
    separable_3b = separable_conv_block(separable_3a,
                                        32,
                                        name='separable_3b'
                                        )

    pooling3 = tf.layers.max_pooling2d(separable_3b,
                                       (2, 2),  # 核大小 变为原来的 1/2
                                       (2, 2),  # 步长
                                       name='pool3'
                                       )
    flatten = tf.contrib.layers.flatten(pooling3)
    logits = tf.layers.dense(flatten, hparams.num_classes)
    #logits = _fully_connected(pooling3, hparams.num_classes)
    labels = features["labels"]

    # 使用交叉熵 设置损失函数
    cost = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # 该api,做了三件事儿 1. y_ -> softmax 2. y -> one_hot 3. loss = ylogy

    # 预测值 获得的是 每一行上 最大值的 索引.注意:tf.argmax()的用法,其实和 np.argmax() 一样的
    predict = tf.nn.softmax(logits,name="softmax_tensor")
    # 将布尔值转化为int类型,也就是 0 或者 1, 然后再和真实值进行比较. tf.equal() 返回值是布尔类型
    #correct_prediction = tf.equal(predict, labels)
    # 比如说第一行最大值索引是6,说明是第六个分类.而y正好也是6,说明预测正确



    # 将上句的布尔类型 转化为 浮点类型,然后进行求平均值,实际上就是求出了准确率
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    if mode in [ModeKeys.PREDICT, ModeKeys.ATTACK]:

      return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'classes': tf.argmax(predict, axis=1),
                'logits': logits,
                'probabilities': predict,
            })

    # Summaries
    # ========================
    tf.summary.scalar("total_nonzero", model_utils.nonzero_count())
    # ========================

    return model_utils.model_top(labels, predict, cost, lr, mode, hparams)

  return mobilenetv2
