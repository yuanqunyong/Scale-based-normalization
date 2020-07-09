#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from batch_norm import batch_norm
from layers import convolution_layer
from layers import full_connection_layer
from hyperparameter import FLAGS

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def conv_bn_relu(inputTensor, shape, bn_param, device):
  conv = convolution_layer(inputTensor, shape, strides=[1,1,1,1], bias=False, layer_name='conv', device=device)
  bn = batch_norm(conv, bn_param=bn_param, scale=False, device = device)
  return tf.nn.relu(bn)
def conv_relu(inputTensor, shape, device):
  conv = convolution_layer(inputTensor, shape, strides=[1,1,1,1], bias=False, layer_name='conv', device=device)
  return tf.nn.relu(conv)


def fc_bn_relu(inputTensor, shape, layer_name, bn_param, device):
  fc = full_connection_layer(inputTensor, shape, bias=False, layer_name=layer_name, device=device)
  bn = batch_norm(fc, bn_param=bn_param, scale=False, name=layer_name, device=device)
  return tf.nn.relu(bn)
def fc_relu(inputTensor, shape, layer_name, device):
  fc = full_connection_layer(inputTensor, shape, bias=False, layer_name=layer_name, device=device)

  return tf.nn.relu(fc)


# def vgg(images, bn_param, keep_prob, depth, num_classes, device=None):
#   def Group(inputTensor, nInputPlane, nOutputPlane, N, name):
#     assert inputTensor.get_shape().as_list()[-1]==nInputPlane
#     layer = [inputTensor]
#     for i in range(N):
#       with tf.variable_scope(name+'_%d'%i):
#         layer += [conv_bn_relu(layer[-1], [3, 3, layer[-1].get_shape().as_list()[-1], nOutputPlane], bn_param, device)]
#
#     layer += [max_pool_2x2(layer[-1])]
#
#     if FLAGS.keep_prob!=None:
#       layer.append(tf.nn.dropout(layer[-1], keep_prob[0]))
#
#     return layer[-1]
#
#   assert len(depth)==5
#
#   layer = [Group(images,      3,  64, depth[0], 'GROUP0')]
#   layer += [Group(layer[-1],  64, 128, depth[1], 'GROUP1')]
#   layer += [Group(layer[-1], 128, 256, depth[2], 'GROUP2')]
#   layer += [Group(layer[-1], 256, 512, depth[3], 'GROUP3')]
#   layer += [Group(layer[-1], 512, 512, depth[4], 'GROUP4')]
#
#   s = layer[-1].get_shape().as_list()
#   layer += [tf.reshape(layer[-1], [-1, np.prod(s[1:])])]
#
#   layer += [fc_bn_relu(layer[-1], [layer[-1].get_shape()[-1], 512], 'fc0', bn_param, device)]
#   layer += [fc_bn_relu(layer[-1], [512, 512], 'fc1', bn_param, device)]
#   layer += [full_connection_layer(layer[-1], [512, num_classes], bias=True, layer_name='fc2', device=device)]
#
#   return layer[-1]
def vgg_common(images, bn_param, keep_prob, depth, num_classes, device=None):
  def Group(inputTensor, nInputPlane, nOutputPlane, N, name):
    assert inputTensor.get_shape().as_list()[-1]==nInputPlane
    layer = [inputTensor]
    for i in range(N):
      with tf.variable_scope(name+'_%d'%i):
        layer += [conv_relu(layer[-1], [3, 3, layer[-1].get_shape().as_list()[-1], nOutputPlane], device)]

    layer += [max_pool_2x2(layer[-1])]

    if FLAGS.keep_prob!=None:
      layer.append(tf.nn.dropout(layer[-1], keep_prob[0]))

    return layer[-1]

  assert len(depth)==5

  layer = [Group(images,      3,  64, depth[0], 'GROUP0')]
  layer += [Group(layer[-1],  64, 128, depth[1], 'GROUP1')]
  layer += [Group(layer[-1], 128, 256, depth[2], 'GROUP2')]
  layer += [Group(layer[-1], 256, 512, depth[3], 'GROUP3')]
  layer += [Group(layer[-1], 512, 512, depth[4], 'GROUP4')]
  layer+=[ batch_norm(layer[-1], bn_param=bn_param, scale=False, device = device)]

  s = layer[-1].get_shape().as_list()
  layer += [tf.reshape(layer[-1], [-1, np.prod(s[1:])])]

  layer += [fc_relu(layer[-1], [layer[-1].get_shape()[-1], 512], 'fc0', device)]
  layer += [fc_relu(layer[-1], [512, 512], 'fc1',  device)]
  layer += [full_connection_layer(layer[-1], [512, num_classes], bias=True, layer_name='fc2', device=device)]

  return layer[-1]
# image_placeholder = tf.placeholder(dtype=tf.float32,
#                               shape=[None, FLAGS.crop_size, FLAGS.crop_size, 3],
#                               name='image_placeholder')
# label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None],
#                               name='label_placeholder')
#
# # logis=vgg(image_placeholder,[0.0],0.7,[1,1,2,2,2],10)
# def loss(logits, labels):
#
#   labels = tf.cast(labels, tf.int64)
#   cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
#   cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#   return cross_entropy_mean
# # losses =loss(logis,label_placeholder)



# logis1=vgg_common(image_placeholder,[0.7],[1,1,2,2,2],10)
# losses1 = loss(logis1, label_placeholder)
# opt = tf.train.MomentumOptimizer(0.01, 0.9)
# grads1 = opt.compute_gradients(losses1)
# x=tf.get_default_graph()
# weight = [i for i in tf.trainable_variables() if 'weight' in i.name]
# yy=x.get_operations()
# print(yy)
# print(x)
# y=tf.trainable_variables()
# print(y)
# train_writer = tf.summary.FileWriter('/tmp/tmp2', graph=x)
# train_writer.add_graph(tf.get_default_graph())
# for i in range(5):
#
#   for var in weight:
#     if ('conv' in var.name) and ('GROUP' + str(i) in var.name):
#       tf.add_to_collection('GROUP' + str(i), var)
#
# for i in  range(5):
#   print(tf.get_collection('GROUP' + str(i)))
