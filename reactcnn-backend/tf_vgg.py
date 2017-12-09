from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.core import dense, dropout
from tensorflow.python.layers.pooling import max_pooling2d
from tensorflow.contrib.layers import flatten, batch_norm
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops

import tensorflow as tf
VGG_ORIGIN_DEPS = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
VGG_CONV_NAMES = ['block1_conv1','block1_conv2',
                  'block2_conv1','block2_conv2',
                  'block3_conv1','block3_conv2','block3_conv3',
                  'block4_conv1', 'block4_conv2', 'block4_conv3',
                  'block5_conv1', 'block5_conv2', 'block5_conv3',]
VGG_FC_NAMES = ['fc1', 'fc2', 'error']
VGG_FC_OUTS = [4096, 4096, 1000]

VFS_FC_NAMES = ['fc1', 'error']
VFS_FC_OUTS = [512, 10]

class ModelBuilder(object):
    def __init__(self, training):
        self.training = training

    def build(self):
        pass

class VGGBuilder(ModelBuilder):

    def __init__(self, training, deps=VGG_ORIGIN_DEPS, conv_names=VGG_CONV_NAMES, fc_names=VGG_FC_NAMES, fc_outs=VGG_FC_OUTS):
        super(VGGBuilder, self).__init__(training=training)
        assert len(deps) == len(conv_names)
        assert len(fc_names) == len(fc_outs)
        self.training = training
        self.deps = deps
        self.fc_outs = fc_outs
        self.conv_names = conv_names
        self.fc_names = fc_names

    def _fc(self, idx, bottom):
        if idx == len(self.fc_outs) - 1:    # last layer
            activation = None
        else:
            activation = nn.relu
        return dense(bottom, self.fc_outs[idx], activation=activation, use_bias=True,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name=self.fc_names[idx])

    def _conv(self, idx, bottom):
        return conv2d(inputs=bottom, filters=self.deps[idx], kernel_size=[3,3], strides=[1,1], padding='same',
            activation=nn.relu, use_bias=True, name=self.conv_names[idx])

    def _conv_with_bn(self, idx, bottom):
        # initializer_dict = {'gamma': init_ops.glorot_normal_initializer()}
        conved = conv2d(inputs=bottom, filters=self.deps[idx], kernel_size=[3,3], strides=[1,1], padding='same',
            activation=None, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=self.conv_names[idx])
        bn = batch_norm(inputs=conved, decay=0.99, center=True, scale=True, activation_fn=tf.nn.relu, is_training=self.training, scope=self.conv_names[idx])
        # initializer_dict = {'gamma': init_ops.glorot_normal_initializer()}
        # conved = conv2d(inputs=bottom, filters=self.deps[idx], kernel_size=[3,3], strides=[1,1], padding='same',
        #     activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=self.conv_names[idx])
        # bn = batch_norm(inputs=conved, center=True, scale=True, activation_fn=None, is_training=self.training, scope=self.conv_names[idx],
        #                 param_initializers=initializer_dict)
        return bn

    def _conv_with_bn_raw(self, idx, bottom):
        conved = conv2d(inputs=bottom, filters=self.deps[idx], kernel_size=[3,3], strides=[1,1], padding='same',
            activation=None, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=self.conv_names[idx])
        bn = batch_norm(inputs=conved, decay=0.99, center=True, scale=True, activation_fn=None, is_training=self.training, scope=self.conv_names[idx])
        return bn

    def _conv_raw(self, idx, bottom):
        return conv2d(inputs=bottom, filters=self.deps[idx], kernel_size=[3, 3], strides=[1, 1], padding='same',
            activation=None, use_bias=True, name=self.conv_names[idx])

    def _maxpool(self, bottom):
        return max_pooling2d(bottom, [2,2], [2,2])

    def _flatten(self, bottom):
        return flatten(bottom)

    def _dropout(self, bottom, drop_rate):
        return dropout(bottom, rate=drop_rate, training=self.training)

    def _relu(self, bottom):
        return tf.nn.relu(bottom)

    # from input to logits
    def build(self, input):
        x = self._conv(0, input)
        x = self._conv(1, x)
        x = self._maxpool(x)
        x = self._conv(2, x)
        x = self._conv(3, x)
        x = self._maxpool(x)
        x = self._conv(4, x)
        x = self._conv(5, x)
        x = self._conv(6, x)
        x = self._maxpool(x)
        x = self._conv(7, x)
        x = self._conv(8, x)
        x = self._conv(9, x)
        x = self._maxpool(x)
        x = self._conv(10, x)
        x = self._conv(11, x)
        x = self._conv(12, x)
        x = self._maxpool(x)
        x = self._flatten(x)
        x = self._fc(0, x)
        x = self._dropout(x, 0.5)
        x = self._fc(1, x)
        x = self._dropout(x, 0.5)
        x = self._fc(2, x)
        return x

    def build_with_bn(self, input):
        x = self._conv_with_bn(0, input)
        x = self._conv_with_bn(1, x)
        x = self._maxpool(x)
        x = self._conv_with_bn(2, x)
        x = self._conv_with_bn(3, x)
        x = self._maxpool(x)
        x = self._conv_with_bn(4, x)
        x = self._conv_with_bn(5, x)
        x = self._conv_with_bn(6, x)
        x = self._maxpool(x)
        x = self._conv_with_bn(7, x)
        x = self._conv_with_bn(8, x)
        x = self._conv_with_bn(9, x)
        x = self._maxpool(x)
        x = self._conv_with_bn(10, x)
        x = self._conv_with_bn(11, x)
        x = self._conv_with_bn(12, x)
        x = self._maxpool(x)
        x = self._flatten(x)
        x = self._fc(0, x)
        x = self._dropout(x, 0.5)
        x = self._fc(1, x)
        x = self._dropout(x, 0.5)
        x = self._fc(2, x)
        return x

class VFSBuilder(VGGBuilder):

    def __init__(self, training, deps=VGG_ORIGIN_DEPS):
        super(VFSBuilder, self).__init__(training, deps, conv_names=VGG_CONV_NAMES, fc_names=VFS_FC_NAMES, fc_outs=VFS_FC_OUTS)

    def build(self, input):
        assert False

    def build_with_bn(self, input):
        x = self._conv_with_bn(0, input)
        x = self._conv_with_bn(1, x)
        x = self._maxpool(x)
        x = self._conv_with_bn(2, x)
        x = self._conv_with_bn(3, x)
        x = self._maxpool(x)
        x = self._conv_with_bn(4, x)
        x = self._conv_with_bn(5, x)
        x = self._conv_with_bn(6, x)
        x = self._maxpool(x)
        x = self._conv_with_bn(7, x)
        x = self._conv_with_bn(8, x)
        x = self._conv_with_bn(9, x)
        x = self._maxpool(x)
        x = self._conv_with_bn(10, x)
        x = self._conv_with_bn(11, x)
        x = self._conv_with_bn(12, x)
        x = self._maxpool(x)
        x = self._flatten(x)
        x = self._fc(0, x)
        x = self._fc(1, x)
        return x

class VFSFullSurveyBuilder(VFSBuilder):

    def __init__(self, training, deps=VGG_ORIGIN_DEPS):
        super(VFSFullSurveyBuilder, self).__init__(training, deps)

    def build_full_outs(self, input):
        result = []
        x = self._conv_with_bn_raw(0, input)
        result.append(x)
        x = self._relu(x)
        x = self._conv_with_bn_raw(1, x)
        result.append(x)
        x = self._relu(x)
        x = self._maxpool(x)
        x = self._conv_with_bn_raw(2, x)
        result.append(x)
        x = self._relu(x)
        x = self._conv_with_bn_raw(3, x)
        result.append(x)
        x = self._relu(x)
        x = self._maxpool(x)
        x = self._conv_with_bn_raw(4, x)
        result.append(x)
        x = self._relu(x)
        x = self._conv_with_bn_raw(5, x)
        result.append(x)
        x = self._relu(x)
        x = self._conv_with_bn_raw(6, x)
        result.append(x)
        x = self._relu(x)
        x = self._maxpool(x)
        x = self._conv_with_bn_raw(7, x)
        result.append(x)
        x = self._relu(x)
        x = self._conv_with_bn_raw(8, x)
        result.append(x)
        x = self._relu(x)
        x = self._conv_with_bn_raw(9, x)
        result.append(x)
        x = self._relu(x)
        x = self._maxpool(x)
        x = self._conv_with_bn_raw(10, x)
        result.append(x)
        x = self._relu(x)
        x = self._conv_with_bn_raw(11, x)
        result.append(x)
        x = self._relu(x)
        x = self._conv_with_bn_raw(12, x)
        result.append(x)
        return result