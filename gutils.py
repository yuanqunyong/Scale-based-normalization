#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
import numpy as np

def norm(v):
  dim = len(v.get_shape())
  return tf.sqrt(tf.reduce_sum(v**2, axis=[i for i in range(dim-1)], keep_dims=True))

def unit(v, eps=1e-8):
  vnorm = norm(v)
  return v/(vnorm+eps), vnorm

def xTy(x, y):
  dim = len(y.get_shape())
  xTy = tf.reduce_sum(x*y, axis=[i for i in range(dim-1)], keep_dims=True, name="xTy")
  return xTy

def clip_by_norm(v, clip_norm):
  dim = len(v.get_shape())
  return tf.clip_by_norm(v, clip_norm, axes=[i for i in range(dim-1)])

def gproj(y, g, normalize=False):
  # implementation of Eq.(6)
  if normalize:
    y,_ = unit(y)

  yTg = xTy(y,g)
  return  g-(yTg*y)

def gexp(y, h, normalize=False):
  # implementation of Eq.(7)
  if normalize:
    y,_ = unit(y)
    h = gproj(y,h)

  u, hnorm = unit(h)
  return y*tf.cos(hnorm) + u*tf.sin(hnorm)

def gpt2(y, h1, h2, normalize=False):
  # implementation of Eq.(8)
  # parallel translation of tangent vector h1 toward h2
  if normalize:
    h1 = gproj(y, h1)
    h2 = gproj(y, h2)

  # svd(h2) = u * sigma * 1
  [u, unorm] = unit(h2)
  uTh1 = xTy(u,h1)
  return h1 - uTh1*( tf.sin(unorm)*y + (1-tf.cos(unorm))*u )

def gpt(y, h, normalize=False):
  # implementation of Eq.(9)

  if normalize:
    h = gproj(y, h)

  [u, unorm] = unit(h)
  return (u*tf.cos(unorm) - y*tf.sin(unorm))*unorm


class unit_initializer(init_ops.Initializer):
  def __init__(self, seed=None, dtype=tf.float32, eps=1e-8):
    self.seed = seed
    self.dtype = dtype
    self.eps = eps

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype

    v = random_ops.truncated_normal(shape, 0, 1.0, dtype, seed=self.seed)
    
    dim = len(v.get_shape())
    vnorm = tf.sqrt(tf.reduce_sum(v**2, axis=[i for i in range(dim-1)], keep_dims=True))

    return v/(vnorm+self.eps)


def conv_norm(conv_weight):
  #    shape_last_dim=conv_weight.shape.as_list()
  # print(conv_weight)
  weight = conv_weight.reshape((-1, conv_weight.shape[-1]))

  norm = np.linalg.norm(weight, axis=0)
  # print(norm)
  return norm

def conv_norm_div(conv_weight):
  #    shape_last_dim=conv_weight.shape.as_list()
  print(conv_weight)
  weight = conv_weight.reshape((-1, conv_weight.shape[-1]))
  shape= weight.shape
  norm = np.linalg.norm(weight, axis=0)/shape[0]
  print(norm)
  return norm

def conv_adjust(conv_weight_value, mul_or_div):
  norm=conv_norm(conv_weight_value)
  for i, item in enumerate(norm):
    if item <= 1e-10 or np.isnan(item):
      norm[i] = 1
  temp = conv_weight_value.shape
  if mul_or_div == True:
    for i in np.arange(temp[-2]):
      conv_weight_value[:, :, i, :] = conv_weight_value[:, :, i, :] * norm[i]
  else:
    for i in np.arange(temp[-1]):
      conv_weight_value[:, :, :, i] = conv_weight_value[:, :, :, i] / norm[i]
  return conv_weight_value

def conv_adjust(conv_weight_value, mul_or_div):
  norm=conv_norm(conv_weight_value)
  for i, item in enumerate(norm):
    if item <= 1e-10 or np.isnan(item):
      norm[i] = 1
  temp = conv_weight_value.shape
  if mul_or_div == True:
    for i in np.arange(temp[-2]):
      conv_weight_value[:, :, i, :] = conv_weight_value[:, :, i, :] * norm[i]
  else:
    for i in np.arange(temp[-1]):
      conv_weight_value[:, :, :, i] = conv_weight_value[:, :, :, i] / norm[i]
  return conv_weight_value


def conv_scal(conv1,conv2):
  temp1 = conv1.shape

  norm=conv_norm(conv1)

  for i, item in enumerate(norm):
    if item <= 1e-1 :
      norm[i] = 0.1
    elif item >=1e+1:
      norm[i]=10
  print(norm)
  for i in range(temp1[-1]):
    conv1[:, :, :, i] = conv1[:, :, :, i] / norm[i]

    conv2[:, :, i, :] = conv2[:, :, i, :] * norm[i]
  return conv1, conv2

def conv_scal_activation(conv1,conv2,scaling_value):
  temp1 = conv1.shape
  temp2 = conv2.shape

  for i, item in enumerate(scaling_value):
    if item <= 1e-10 or np.isnan(item):
      scaling_value[i] = 1
  # print(norm)
  for i in range(temp1[-1]):
    conv1[:, :, :, i] = conv1[:, :, :, i] / scaling_value[i]
  for k in range(temp2[-2]):
    conv2[:, :, i, :] = conv2[:, :, i, :] * scaling_value[i]
  return conv1, conv2

def conv_scal_select(conv1,conv2,scaling_value):

  # print(norm)
  for i,item in enumerate(scaling_value):
    if item > 1:
      conv1[:, :, :, i] = conv1[:, :, :, i] / scaling_value[i]
      conv2[:, :, i, :] = conv2[:, :, i, :] * scaling_value[i]
  return conv1, conv2

def last_conv_scal_select(conv1,scaling_value):

  # print(norm)
  for i,item in enumerate(scaling_value):
    if item > 1:
      conv1[:, :, :, i] = conv1[:, :, :, i] / scaling_value[i]

  return conv1


def conv_scal_div(conv1,conv2):
  temp1 = conv1.shape
  temp2 = conv2.shape
  norm=conv_norm_div(conv1)

  for i, item in enumerate(norm):
    if abs(item )<= 1e-10 or np.isnan(item):
      norm[i] = 1
  print(norm)
  for i in range(temp1[-1]):
    conv1[:, :, :, i] = conv1[:, :, :, i] / norm[i]
  for k in range(temp2[-2]):
    conv2[:, :, i, :] = conv2[:, :, i, :] * norm[i]
  return conv1, conv2


def last_conv_scal(conv):
  temp=conv.shape
  norm=conv_norm(conv)
  for i in range(temp[-1]):
    conv[:, :, :, i] = conv[:, :, :, i] / norm[i]

  return conv

def last_conv_scal_div(conv):
  temp=conv.shape
  norm=conv_norm_div(conv)
  for i in range(temp[-1]):
    conv[:, :, :, i] = conv[:, :, :, i] / norm[i]

  return conv

def last_conv_scal_activation(conv, scaling_value):
  temp=conv.shape
  for i, item in enumerate(scaling_value):
    if abs(item )<= 1e-10 or np.isnan(item):
      conv[i] = 1
  for i in range(temp[-1]):
    conv[:, :, :, i] = conv[:, :, :, i] / scaling_value[i]

  return conv


def computer_between_layer(conv_block_list,mean_value_list):
  activation_mean_list=[]
  mean_geometry=1
  list_leth=len(mean_value_list)
  for i in range(list_leth):
    temp=np.mean(mean_value_list[i])
    mean_geometry=mean_geometry*temp
    activation_mean_list.append(temp)
  mean_geometry=mean_geometry**(1/list_leth)
  for i in range(list_leth):
    conv_block_list[i]=conv_block_list[i]*(activation_mean_list[i]/mean_geometry)

  return conv_block_list