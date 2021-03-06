# encoding: utf-8
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import preprocessing
import grassmann_optimizer
import models
import gutils

from hyperparameter import FLAGS
import pdb

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


# def compute_activation(pre_activation):
#
#   pre_activation1 = sess.run(pre_activation.outputs, feed_dict=feed_dict)
#
#   return pre_activation1


def backward_scaling_adjust_grad_nobias(layer_gra, layer_weight):
    lenth = len(layer_weight)

    temp = 100 ** lenth
    layer_mean = []
    for i in range(lenth):
        weight = layer_weight[i]
        shape = weight.shape
        weight = np.reshape(weight, (-1, shape[-1]))
        norm_2 = np.linalg.norm(weight)
        print(norm_2)
        mean_temp = np.mean(np.square(layer_gra[i] / norm_2))
        # mean_temp=np.sqrt(mean_temp)
        print(mean_temp)
        layer_mean.append(mean_temp)
        temp = temp * mean_temp
    temp = (temp / 100 ** lenth) ** (1 / lenth)
    print(temp)
    print(100 * '*')
    for i, item in enumerate(layer_mean):
        if item <= 1e-15 or np.isnan(item):
            layer_mean[i] = 1
        print(np.sqrt((layer_mean[i] / temp)))
        layer_weight[i] = layer_weight[i] * np.sqrt((layer_mean[i] / temp))
        # print(layer_weight[i])

    return layer_weight
    #    print(mean)
    # meang = 1
    # for x in mean:
    #   meang = meang * x
    # meang = meang ** (1 / 3)
    # if abs(meang) <= 1e-15 or np.isnan(meang):
    #   return None

    # mean = [meang / mean[0], meang / mean[1], meang / mean[2]]
    # print(mean)
    # a_weigh = a * mean[0]
    # b_weigh = b * mean[1]
    # c_weigh = c * mean[2]
    #
    # return a_weigh, b_weigh, c_weigh


def backward_scaling_adjust_grad_mean(layer_gra, layer_weight):
    lenth = len(layer_weight)
    temp = 100 ** lenth
    layer_mean = []
    for i in range(lenth):
        weight = np.abs(layer_weight[i])
        gra = np.abs(layer_gra[i])
        mean_temp = np.mean(gra / weight)
        print(mean_temp)
        layer_mean.append(mean_temp)
        temp = temp * mean_temp
    temp = (temp / 100 ** lenth) ** (1 / lenth)
    print(temp)

    for i, item in enumerate(layer_mean):
        if item <= 1e-15 or np.isnan(item):
            layer_mean[i] = 1
        layer_weight[i] = layer_weight[i] * (layer_mean[i] / temp)

    return layer_weight


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def loss(logits, labels):
    '''
    Calculate the cross entropy loss given logits and true labels
    :param logits: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size]
    :return: loss tensor with shape [1]
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


def top_k_right(predictions, labels, k):
    '''
    Calculate the top-k error
    :param predictions: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size, 1]
    :param k: int
    :return: tensor with shape [1]
    '''
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    num_correct = tf.reduce_sum(in_top1)
    return num_correct


class Trainer(object):
    '''
    This Object is responsible for all the training and validation process
    '''

    def __init__(self):
        self.global_step = []
        self.batch_queue = np.array([], np.int64)

        self.device_main = '/gpu:0'
        self.device_model = '/gpu:0'
        self.device_opt = '/gpu:0'

        self.decay_step = []
        self.max_step = []
        self.num_classes = []

    def print_params(self):
        logging.info('data: %s' % FLAGS.data)
        logging.info('task: %s' % FLAGS.task)
        logging.info('model: %s' % FLAGS.model)
        if FLAGS.model == 'resnet' or FLAGS.model == 'resnet_sc_bn' or FLAGS.model == 'resnet_sc':
            logging.info('depth: %d, widen_factor: %d' % (FLAGS.depth, FLAGS.widen_factor))

        logging.info('optimizer: %s' % (FLAGS.optimizer))
        logging.info('[weightDecay, biasDecay, gammaDecay, betaDecay]=[%s, %s, %s, %s]' % (
        FLAGS.weightDecay, FLAGS.biasDecay, FLAGS.gammaDecay, FLAGS.betaDecay))
        logging.info(
            '[learnRateE, learnRateG, momentum]=[%s, %s, %s]' % (FLAGS.learnRate, FLAGS.learnRateG, FLAGS.momentum))
        logging.info('learnRateDecay: %f' % FLAGS.learnRateDecay)
        logging.info('[omega, grad_clip]=[%s, %s]' % (FLAGS.omega, FLAGS.grad_clip))

        if FLAGS.data == 'cifar10' or FLAGS.data == 'cifar100':
            decay_epoch = [60, 120, 160]
            max_epoch = 200
            ndata = 50000
        elif FLAGS.data == 'svhn':
            decay_epoch = [60, 120]
            max_epoch = 160
            ndata = 604388
        else:
            assert False, 'unkown data'
        if FLAGS.task == 'train_scaling':
            logging.info('scal_freq: %s' % (FLAGS.scal_freq))

        self.decay_step = [i * ndata // FLAGS.batch_size for i in decay_epoch]
        self.max_step = max_epoch * ndata // FLAGS.batch_size

        logging.info('batch_size: %d' % FLAGS.batch_size)
        logging.info('num_gpus: %d' % FLAGS.num_gpus)
        logging.info('decay_epoch: %s, decay_step: %s' % (decay_epoch, self.decay_step))
        logging.info('max_epoch: %s, max_step: %s' % (max_epoch, self.max_step))
        logging.info('keep_prob: %s' % FLAGS.keep_prob)
        logging.info('crop_size: %d' % FLAGS.crop_size)
        logging.info('padding_size: %d' % FLAGS.padding_size)
        logging.info('vali_freq: %d' % FLAGS.vali_freq)
        logging.info('save_freq: %d' % FLAGS.save_freq)
        logging.info('max_to_keep %d' % (FLAGS.max_to_keep))
        logging.info('load: %s' % FLAGS.load)
        #    logging.info('num_residual_block %d, wide_Factor %d'%(FLAGS.num_residual_blocks, FLAGS.wide_factor))

        if FLAGS.data == 'cifar10':
            self.num_classes = 10
        elif FLAGS.data == 'cifar100':
            self.num_classes = 100
        elif FLAGS.data == 'svhn':
            self.num_classes = 10

    def placeholders(self):
        self.image_placeholders = []
        self.label_placeholders = []
        self.batch_norm_param_placeholder = tf.placeholder(tf.float32, shape=[1], name='bn_eval')
        self.learn_rate_placeholder = tf.placeholder(tf.float32, shape=[2], name='learn_rate')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, shape=[1], name='keep_prob')

    def initialize_optimizer(self):
        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.zeros_initializer(), trainable=False)

        if FLAGS.optimizer == 'sgd':
            opt = tf.train.MomentumOptimizer(self.learn_rate_placeholder[0], FLAGS.momentum,
                                             use_nesterov=FLAGS.nesterov)
        elif FLAGS.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(self.learn_rate_placeholder[0])
        elif FLAGS.optimizer == 'sgdg':
            opta = tf.train.MomentumOptimizer(self.learn_rate_placeholder[0], FLAGS.momentum,
                                              use_nesterov=FLAGS.nesterov)
            optb = grassmann_optimizer.SgdgOptimizer(self.learn_rate_placeholder[1], FLAGS.momentum,
                                                     grad_clip=FLAGS.grad_clip)
            opt = grassmann_optimizer.HybridOptimizer(opta, optb)

        elif FLAGS.optimizer == 'adamg':
            opta = tf.train.MomentumOptimizer(self.learn_rate_placeholder[0], FLAGS.momentum,
                                              use_nesterov=FLAGS.nesterov)
            optb = grassmann_optimizer.AdamgOptimizer(self.learn_rate_placeholder[1], FLAGS.momentum,
                                                      epsilon=FLAGS.adam_eps, grad_clip=FLAGS.grad_clip)
            opt = grassmann_optimizer.HybridOptimizer(opta, optb)


        else:
            assert False, 'unknown optimizer'

        return opt

    def build_graph_train(self, opt, grads):
        with tf.device(self.device_opt):
            if FLAGS.optimizer == 'sgdg' or FLAGS.optimizer == 'adamg':
                grads_a = [i for i in grads if not i[1] in tf.get_collection('grassmann')]
                grads_b = [i for i in grads if i[1] in tf.get_collection('grassmann')]

                apply_gradient_op = opt.apply_gradients(grads_a, grads_b, global_step=self.global_step)
            else:
                apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

        return tf.group(*([apply_gradient_op] + tf.get_collection('batch_norm_update')))

    def build_graph_model(self, opt):
        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_losses = []
        tower_right = []
        tower_k_right = []
        tower_cross_entropy = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        image_placeholder = tf.placeholder(dtype=tf.float32,
                                                           shape=[None, FLAGS.crop_size, FLAGS.crop_size, 3],
                                                           name='image_placeholder')
                        label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None],
                                                           name='label_placeholder')

                        self.image_placeholders.append(image_placeholder)
                        self.label_placeholders.append(label_placeholder)

                        # Build inference Graph.
                        logits = models.model(image_placeholder,
                                              bn_param=self.batch_norm_param_placeholder,
                                              keep_prob=self.keep_prob_placeholder,
                                              num_classes=self.num_classes,
                                              device=self.device_model,
                                              FLAGS=FLAGS)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        cross_entropy_mean = loss(logits, label_placeholder)
                        # Assemble all of the losses for the current tower only.


                        weight = [i for i in tf.trainable_variables() if 'weight' in i.name]
                        bias = [i for i in tf.trainable_variables() if 'bias' in i.name]
                        beta = [i for i in tf.trainable_variables() if 'beta' in i.name]
                        gamma = [i for i in tf.trainable_variables() if 'gamma' in i.name]

                        assert len(weight) + len(bias) + len(beta) + len(gamma) == len(tf.trainable_variables())

                        if i == 0 and FLAGS.grassmann:
                            for var in weight:
                                undercomplete = np.prod(var.shape[0:-1]) > var.shape[-1]
                                if undercomplete and ('conv' in var.name):
                                    ## initialize to scale 1
                                    var._initializer_op = tf.assign(var, gutils.unit_initializer()(var.shape)).op
                                    tf.add_to_collection('grassmann', var)

                        ## build graphs for regularization
                        if FLAGS.omega is not None:
                            for var in tf.get_collection('grassmann'):
                                shape = var.get_shape().as_list()
                                v = tf.reshape(var, [-1, shape[-1]])
                                v_sim = tf.matmul(tf.transpose(v), v)

                                eye = tf.eye(shape[-1])
                                assert eye.get_shape() == v_sim.get_shape()

                                orthogonality = tf.multiply(tf.reduce_sum((v_sim - eye) ** 2), 0.5 * FLAGS.omega,
                                                            name='orthogonality')
                                tf.add_to_collection('orthogonality', orthogonality)

                        if FLAGS.weightDecay is not None:
                            for var in [i for i in weight if not i in tf.get_collection('grassmann')]:
                                weight_decay = tf.multiply(tf.nn.l2_loss(var), FLAGS.weightDecay, name='weightcost')
                                tf.add_to_collection('weightcost', weight_decay)

                        if FLAGS.biasDecay is not None:
                            for var in bias:
                                weight_decay = tf.multiply(tf.nn.l2_loss(var), FLAGS.biasDecay, name='weightcost')
                                tf.add_to_collection('weightcost', weight_decay)

                        if FLAGS.gammaDecay is not None:
                            for var in gamma:
                                weight_decay = tf.multiply(tf.nn.l2_loss(var), FLAGS.gammaDecay, name='weightcost')
                                tf.add_to_collection('weightcost', weight_decay)

                        if FLAGS.betaDecay is not None:
                            for var in beta:
                                weight_decay = tf.multiply(tf.nn.l2_loss(var), FLAGS.betaDecay, name='weightcost')
                                tf.add_to_collection('weightcost', weight_decay)

                        if tf.get_collection('weightcost', scope):
                            weightcost = tf.add_n(tf.get_collection('weightcost', scope), name='weightcost')
                        else:
                            weightcost = tf.zeros([1])

                        if tf.get_collection('orthogonality', scope):
                            orthogonality = tf.add_n(tf.get_collection('orthogonality', scope), name='orthogonality')
                        else:
                            orthogonality = tf.zeros([1])

                        if FLAGS.task == 'train_scaling':
                            if '_c' in FLAGS.model:
                                # for i in range(5):
                                #
                                #    for var in weight:
                                #      if ('conv' in var.name) and ('GROUP'+str(i) in var.name):
                                #        tf.add_to_collection('GROUP'+str(i),var)

                                var_con = tf.trainable_variables()

                            elif FLAGS.model == 'resnet':
                                for var in weight:
                                    if ('conv' in var.name) and ('in_block' in var.name):
                                        tf.add_to_collection('convultion', var)
                                var_con = tf.get_collection('convolution')


                            elif FLAGS.model == 'resnet_sc' or FLAGS.model == 'resnet_sc_bn':
                                n = (FLAGS.depth - 4) // 6
                                var_con = []
                                for i in range(2, 5):
                                    for k in range(n):
                                        for var in weight:
                                            if ('conv' in var.name) and (
                                                            'group' + str(i) + '_block' + str(k) in var.name):
                                                tf.add_to_collection('group' + str(i) + '_block' + str(k), var)
                                        var_con.append(tf.get_collection('group' + str(i) + '_block' + str(k)))
                            else:

                                assert False, 'unknown model'

                        # Calculate the total loss for the current tower.
                        total_loss = cross_entropy_mean + weightcost + orthogonality
                        t_1_right = top_k_right(logits, label_placeholder, 1)
                        t_k_right = top_k_right(logits, label_placeholder, FLAGS.k)

                        if opt != None:
                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads = opt.compute_gradients(total_loss)
                            tower_grads.append(grads)

                        tower_losses.append(total_loss)
                        tower_right.append(t_1_right)
                        tower_k_right.append(t_k_right)
                        tower_cross_entropy.append(cross_entropy_mean)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        average_grads = average_gradients(tower_grads)

        losses = tf.reduce_mean(tower_losses)
        right = tf.reduce_sum(tower_right)
        k_right = tf.reduce_sum(tower_k_right)
        cross_entropy = tf.reduce_mean(tower_cross_entropy, name='cross_entropy')
        if FLAGS.task == 'train_scaling':

            return average_grads, losses, right, k_right, cross_entropy, var_con, logits
        else:
            return average_grads, losses, right, k_right, cross_entropy

    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''

        num_data = len(train_labels)

        if len(self.batch_queue) < train_batch_size:
            self.batch_queue = np.concatenate((self.batch_queue, np.random.permutation(num_data)))

        idx = self.batch_queue[0:train_batch_size]
        self.batch_queue = self.batch_queue[train_batch_size:]

        batch_data = train_data[idx, ...]

        if FLAGS.data == 'cifar10' or FLAGS.data == 'cifar100':
            batch_data = preprocessing.random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)
        elif FLAGS.data == 'svhn':
            batch_data = batch_data.astype(np.float32) / 255.

        batch_label = train_labels[idx]

        return batch_data, batch_label

    def full_validation(self, tensor, sess, data, labels, bn_eval=1.0):
        feed_dict = {self.learn_rate_placeholder: [0.0, 0.0],
                     self.batch_norm_param_placeholder: [bn_eval],
                     self.keep_prob_placeholder: [1.0]}

        value_accum = np.zeros_like(tensor)

        data_per_tower = FLAGS.vali_batch_size // FLAGS.num_gpus
        assert data_per_tower * FLAGS.num_gpus == FLAGS.vali_batch_size, 'vali_batch_size should be multiple of num_gpus'
        num_batches = len(labels) // FLAGS.vali_batch_size

        sidx = 0
        eidx = data_per_tower
        for _ in range(num_batches):
            for i in range(FLAGS.num_gpus):
                feed_dict.update({self.image_placeholders[i]: data[sidx:eidx, ...],
                                  self.label_placeholders[i]: labels[sidx:eidx, ...]})
                sidx += data_per_tower
                eidx += data_per_tower

            value = sess.run(tensor, feed_dict=feed_dict)
            value[0] *= (data_per_tower * FLAGS.num_gpus)
            value_accum += value

        if sidx < len(labels):
            assert (len(labels) - sidx) % FLAGS.num_gpus == 0
            data_per_tower = (len(labels) - sidx) // FLAGS.num_gpus

            eidx = sidx + data_per_tower
            for i in range(FLAGS.num_gpus):
                feed_dict.update({self.image_placeholders[i]: data[sidx:eidx, ...],
                                  self.label_placeholders[i]: labels[sidx:eidx, ...]})
                sidx += data_per_tower
                eidx += data_per_tower

            value = sess.run(tensor, feed_dict=feed_dict)
            value[0] *= FLAGS.vali_batch_size
            value_accum += value

        return value_accum / len(labels)

    def data(self):
        if FLAGS.data == 'cifar10':
            import cifar10_input
            cifar10_input.maybe_download_and_extract()

            all_data, all_labels = cifar10_input.read_train_data()
            vali_data, vali_labels = cifar10_input.read_validation_data()

            all_data_mean = np.mean(all_data, axis=(0, 1, 2))
            all_data_std = np.std(all_data, axis=(0, 1, 2))

            all_data -= all_data_mean
            all_data /= all_data_std

            vali_data -= all_data_mean
            vali_data /= all_data_std

            all_data = preprocessing.pad_images(all_data, FLAGS.padding_size)

        elif FLAGS.data == 'cifar100':
            import cifar100_input
            cifar100_input.maybe_download_and_extract()

            all_data, all_labels = cifar100_input.read_train_data()
            vali_data, vali_labels = cifar100_input.read_validation_data()

            all_data_mean = np.mean(all_data, axis=(0, 1, 2))
            all_data_std = np.std(all_data, axis=(0, 1, 2))

            all_data -= all_data_mean
            all_data /= all_data_std

            vali_data -= all_data_mean
            vali_data /= all_data_std

            all_data = preprocessing.pad_images(all_data, FLAGS.padding_size)

        elif FLAGS.data == 'svhn':
            import svhn_input
            svhn_input.maybe_download()

            all_data, all_labels = svhn_input.read_train_data()
            vali_data, vali_labels = svhn_input.read_test_data()

            # label is [1,...,10], so subtract one to make [0,...,9]
            all_labels -= 1
            vali_labels -= 1

            # To save memory, train_data will be divided by 255 when it feeds.
            vali_data = vali_data.astype(np.float32) / 255.
        else:
            assert False, 'unknown data'

        return all_data, all_labels, vali_data, vali_labels

    def train(self, train_path):
        self.print_params()
        all_data, all_labels, vali_data, vali_labels = self.data()

        with tf.Graph().as_default(), tf.device(self.device_main):
            self.placeholders()
            opt = self.initialize_optimizer()
            grads, losses, right, right_k, cross_entropy = self.build_graph_model(opt)
            train_op = self.build_graph_train(opt, grads)

            train_writer = tf.summary.FileWriter(train_path)
            train_writer.add_graph(tf.get_default_graph())
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.log_device_placement = FLAGS.log_device_placement
            if FLAGS.visible_devices is not None:
                config.gpu_options.visible_device_list = FLAGS.visible_devices

            sess = tf.Session(config=config)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
            if FLAGS.load is None:
                sess.run(tf.global_variables_initializer())
                start_step = 0
            else:
                logging.info('loading ckpt from %s' % FLAGS.load)
                saver.restore(sess, FLAGS.load)
                start_step = int(sess.run(self.global_step))

            # tf.train.start_queue_runners(sess=sess)

            t0 = time.time()
            for step in range(start_step, self.max_step):
                power = sum(step > i for i in self.decay_step)
                lr_e = FLAGS.learnRate * pow(FLAGS.learnRateDecay, power)
                lr_g = FLAGS.learnRateG * pow(FLAGS.learnRateDecay, power)

                feed_dict = {self.learn_rate_placeholder: [lr_e, lr_g],
                             self.batch_norm_param_placeholder: [0.0],
                             self.keep_prob_placeholder: [FLAGS.keep_prob]}

                data_per_tower = FLAGS.batch_size // FLAGS.num_gpus
                assert data_per_tower * FLAGS.num_gpus == FLAGS.batch_size, 'batch_size should be multiple of num_gpus'
                for i in range(FLAGS.num_gpus):
                    train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data,
                                                                                             all_labels, data_per_tower)
                    feed_dict.update({self.image_placeholders[i]: train_batch_data,
                                      self.label_placeholders[i]: train_batch_labels})

                if (step + 1) % FLAGS.report_freq == 0:
                    ops = [train_op, losses, right, right_k, cross_entropy] + \
                          tf.get_collection('orthogonality') + \
                          tf.get_collection('weightcost')
                else:
                    ops = [train_op]

                start_time = time.time()
                values = sess.run(ops, feed_dict=feed_dict)
                duration = time.time() - start_time

                if (step + 1) % FLAGS.report_freq == 0:
                    loss_value = values[1]
                    right_value = values[2]
                    right_k_value = values[3]
                    cross_entropy_value = values[4]

                    s = 5;
                    e = s + len(tf.get_collection('orthogonality'));
                    orthogonality_value = values[s:e]
                    s = e;
                    e = s + len(tf.get_collection('weightcost'));
                    weightcost_value = values[s:e]

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration

                    error_rate = (FLAGS.batch_size - right_value) / FLAGS.batch_size
                    error_k_rate = (FLAGS.batch_size - right_k_value) / FLAGS.batch_size

                    epoch = step * FLAGS.batch_size // len(all_labels)

                    format_str = (
                    '%s: step %d (epoch %d), loss = %f, err = %f %%, err_%d = %f %%, cross_entropy = %f, lr_e = %f, lr_g = %f (%.1f examples/sec; %.3f ''sec/batch)')
                    logging.info(format_str % (datetime.now(), step, epoch,
                                               loss_value, error_rate * 100., FLAGS.k, error_k_rate,
                                               cross_entropy_value,
                                               lr_e, lr_g,
                                               examples_per_sec, sec_per_batch))
                    if FLAGS.verbose:
                        if orthogonality_value:
                            logging.info('orthogonality(%d)=%f, %s' % (
                            len(orthogonality_value), sum(orthogonality_value), orthogonality_value))
                        if weightcost_value:
                            logging.info('weightcost(%d)=%f, %s' % (
                            len(weightcost_value), sum(weightcost_value), weightcost_value))
                        logging.info('')

                if (step + 1) % FLAGS.vali_freq == 0 or (step + 1) == self.max_step:
                    start_time = time.time()
                    values = self.full_validation([cross_entropy, right, right_k], sess, vali_data, vali_labels)
                    duration = time.time() - start_time

                    format_str = (
                    'validation: step %d (epoch %d), cross_entropy = %.4f, err = %.4f %%, err_%d = %.4f %% (%.1f examples/sec; %.3f ''sec)')
                    logging.info(format_str % (
                    step, epoch, values[0], (1. - values[1]) * 100., FLAGS.k, (1. - values[2]) * 100.,
                    len(vali_labels) / duration, duration))
                    print('saving to ' + os.path.join(train_path, 'log.txt'))
                    logging.info('')

                if (step + 1) % FLAGS.save_freq == 0 or (step + 1) == self.max_step:
                    saver.save(sess, os.path.join(train_path, 'model.ckpt'), global_step=step)
                    logging.info('saving to %s/model.ckpt-%d\n' % (train_path, step))

            logging.info('total training time: %d sec' % (time.time() - t0))

        return

    def test(self):
        self.print_params()
        all_data, all_labels, vali_data, vali_labels = self.data()

        with tf.Graph().as_default(), tf.device(self.device_main):
            self.placeholders()
            losses, right, right_k, cross_entropy, _ = self.build_graph_model(opt=None)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.log_device_placement = FLAGS.log_device_placement
            if FLAGS.visible_devices is not None:
                config.gpu_options.visible_device_list = FLAGS.visible_devices

            sess = tf.Session(config=config)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
            if FLAGS.load is None:
                sess.run(tf.global_variables_initializer())
            else:
                logging.info('loading ckpt from %s' % FLAGS.load)
                saver.restore(sess, FLAGS.load)

            start_time = time.time()
            values = self.full_validation([cross_entropy, right, right_k], sess, vali_data, vali_labels)
            duration = time.time() - start_time

            format_str = (
            'validation: cross_entropy = %.4f, err = %.4f %%, err_%d = %.4f %% (%.1f examples/sec; %.3f ''sec)')
            logging.info(format_str % (
            values[0], (1. - values[1]) * 100., FLAGS.k, (1. - values[2]) * 100., len(vali_labels) / duration,
            duration))

        return

    def see(self):
        #    np.set_printoptions(precision=4, threshold=np.inf)
        self.print_params()
        all_data, all_labels, vali_data, vali_labels = self.data()

        with tf.Graph().as_default(), tf.device(self.device_main):
            self.placeholders()
            _, losses, right, right_k, cross_entropy = self.build_graph_model(opt=None)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.log_device_placement = FLAGS.log_device_placement
            if FLAGS.visible_devices is not None:
                config.gpu_options.visible_device_list = FLAGS.visible_devices

            sess = tf.Session(config=config)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
            if FLAGS.load is None:
                sess.run(tf.global_variables_initializer())
            else:
                logging.info('loading ckpt from %s' % FLAGS.load)
                saver.restore(sess, FLAGS.load)

            total_weights = 0
            for v in tf.trainable_variables():
                if v in tf.get_collection('grassmann'):
                    log = 'G) '
                else:
                    log = 'E) '

                nweights = np.prod(v.get_shape().as_list())
                log += '%s %s: %d' % (v.op.name, v.get_shape(), nweights)

                total_weights += nweights

                logging.info(log)
                if FLAGS.verbose:
                    logging.info(sess.run(v))

        logging.info('total weights: %d' % total_weights)
        return

    def train_scaling(self, train_path):
        self.print_params()
        all_data, all_labels, vali_data, vali_labels = self.data()

        with tf.Graph().as_default(), tf.device(self.device_main):
            self.placeholders()
            opt = self.initialize_optimizer()
            grads, losses, right, right_k, cross_entropy, var_con, logist = self.build_graph_model(opt)
            train_op = self.build_graph_train(opt, grads)

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.log_device_placement = FLAGS.log_device_placement
            if FLAGS.visible_devices is not None:
                config.gpu_options.visible_device_list = FLAGS.visible_devices

            sess = tf.Session(config=config)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)
            if FLAGS.load is None:
                sess.run(tf.global_variables_initializer())
                start_step = 0
            else:
                logging.info('loading ckpt from %s' % FLAGS.load)
                saver.restore(sess, FLAGS.load)
                start_step = int(sess.run(self.global_step))

            gl_var = tf.global_variables()
            if FLAGS.optimizer == 'sgd':
                slots = [x for x in gl_var if "Momentum" in x.name]

            elif FLAGS.optimizer == 'adm':
                slots = [x for x in gl_var if "Adam" in x.name]

            # print(train_var)
            for ii in var_con:
                print(ii)
            train_writer = tf.summary.FileWriter(train_path)
            train_writer.add_graph(tf.get_default_graph())

            g = tf.get_default_graph()
            operation = g.get_operations()
            pre_activation = [x for x in operation if ("Conv2D" in x.name) and ('grad' not in x.name)]
            for item in pre_activation:
                print(item.name)

            def computer_var_mean(pre_activation):

                pre_activation_temp = sess.run(pre_activation.outputs, feed_dict=feed_dict)
                temppre = np.abs(pre_activation_temp[0])
                # temppre = pre_activation_temp[0]
                # pre_activation_variance = np.std(temppre, axis=0)
                pre_activation_variance = np.std(temppre, axis=(0, 1, 2))
                pre_activation_mean = np.mean(temppre, axis=(0, 1, 2))
                # pre_activation_mean = np.mean(temppre, axis=0)

                return pre_activation_variance, pre_activation_mean

            t0 = time.time()
            for step in range(start_step, self.max_step):
                power = sum(step > i for i in self.decay_step)
                lr_e = FLAGS.learnRate * pow(FLAGS.learnRateDecay, power)
                lr_g = FLAGS.learnRateG * pow(FLAGS.learnRateDecay, power)

                feed_dict = {self.learn_rate_placeholder: [lr_e, lr_g],
                             self.batch_norm_param_placeholder: [0.0],
                             self.keep_prob_placeholder: [FLAGS.keep_prob]}

                data_per_tower = FLAGS.batch_size // FLAGS.num_gpus
                assert data_per_tower * FLAGS.num_gpus == FLAGS.batch_size, 'batch_size should be multiple of num_gpus'
                for i in range(FLAGS.num_gpus):
                    train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data,
                                                                                             all_labels, data_per_tower)
                    feed_dict.update({self.image_placeholders[i]: train_batch_data,
                                      self.label_placeholders[i]: train_batch_labels})

                if (step + 1) % FLAGS.report_freq == 0:
                    ops = [train_op, losses, right, right_k, cross_entropy] + \
                          tf.get_collection('orthogonality') + \
                          tf.get_collection('weightcost')
                else:
                    ops = [train_op]

                start_time = time.time()
                values = sess.run(ops, feed_dict=feed_dict)
                duration = time.time() - start_time

                if (step + 1) % FLAGS.report_freq == 0:
                    loss_value = values[1]
                    right_value = values[2]
                    right_k_value = values[3]
                    cross_entropy_value = values[4]
                    s = 5;
                    e = s + len(tf.get_collection('orthogonality'));
                    orthogonality_value = values[s:e]
                    s = e;
                    e = s + len(tf.get_collection('weightcost'));
                    weightcost_value = values[s:e]

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration

                    error_rate = (FLAGS.batch_size - right_value) / FLAGS.batch_size
                    error_k_rate = (FLAGS.batch_size - right_k_value) / FLAGS.batch_size

                    epoch = step * FLAGS.batch_size // len(all_labels)

                    format_str = (
                        '%s: step %d (epoch %d), loss = %f, err = %f %%, err_%d = %f %%, cross_entropy = %f, lr_e = %f, lr_g = %f (%.1f examples/sec; %.3f ''sec/batch)')
                    logging.info(format_str % (datetime.now(), step, epoch,
                                               loss_value, error_rate * 100., FLAGS.k, error_k_rate,
                                               cross_entropy_value,
                                               lr_e, lr_g,
                                               examples_per_sec, sec_per_batch))

                    if FLAGS.verbose:
                        if orthogonality_value:
                            logging.info(
                                'orthogonality(%d)=%f, %s' % (
                                len(orthogonality_value), sum(orthogonality_value), orthogonality_value))
                        if weightcost_value:
                            logging.info('weightcost(%d)=%f, %s' % (
                            len(weightcost_value), sum(weightcost_value), weightcost_value))
                        logging.info('')

                if (step + 1) % FLAGS.vali_freq == 0 or (step + 1) == self.max_step:
                    start_time = time.time()
                    values = self.full_validation([cross_entropy, right, right_k], sess, vali_data, vali_labels)
                    duration = time.time() - start_time

                    format_str = (
                        'validation: step %d (epoch %d), cross_entropy = %.4f, err = %.4f %%, err_%d = %.4f %% (%.1f examples/sec; %.3f ''sec)')
                    logging.info(format_str % (
                        step, epoch, values[0], (1. - values[1]) * 100., FLAGS.k, (1. - values[2]) * 100.,
                        len(vali_labels) / duration,
                        duration))
                    print('saving to ' + os.path.join(train_path, 'log.txt'))
                    logging.info('')

                if (step + 1) % FLAGS.save_freq == 0 or (step + 1) == self.max_step:
                    saver.save(sess, os.path.join(train_path, 'model.ckpt'), global_step=step)
                    logging.info('saving to %s/model.ckpt-%d\n' % (train_path, step))

                if (step + 1) % FLAGS.scal_freq == 0:
                    if 'vgg' in FLAGS.model:
                        if FLAGS.model == 'vgg11_c' or FLAGS.model == 'vgg11':
                            depth = [1, 1, 2, 2, 2]
                        elif FLAGS.model == 'vgg13_c' or FLAGS.model == 'vgg13':
                            depth = [2, 2, 2, 2, 2]
                        elif FLAGS.model == 'vgg16_c' or FLAGS.model == 'vgg16':
                            depth = [2, 2, 3, 3, 3]
                        elif FLAGS.model == 'vgg19_c' or FLAGS.model == 'vgg19':
                            depth = [2, 2, 4, 4, 4]

                        totol_conv = sum(depth)

                        values0 = self.full_validation([cross_entropy], sess, vali_data, vali_labels)
                        conv_list = []
                        for k in range(totol_conv):
                            conv_temp = sess.run(var_con[k])
                            conv_list.append(conv_temp)

                        if step > 1 and step <= self.decay_step[0]:

                            for k in range(5):
                                block_depth = depth[k]

                                start_layer=sum(depth[:k])
                                if block_depth > 1:
                                    conv_block_list = []
                                    mean_value_list = []
                                    for i in range(block_depth):

                                        if i < block_depth - 1:
                                            id_layer=start_layer+i
                                            var_value, mean_value = computer_var_mean(pre_activation[id_layer])
                                            mean_value_list.append(mean_value)
                                            print(var_value)
                                            print(mean_value)
                                            conv_weight_1, conv_weight_2 = sess.run([var_con[id_layer], var_con[id_layer+ 1]])
                                            scaling_value = var_value / mean_value
                                            print(scaling_value)
                                            conv1, conv2 = gutils.conv_scal_select(conv_weight_1, conv_weight_2,
                                                                                   scaling_value)
                                            var_con[id_layer].load(conv1, sess)
                                            var_con[id_layer + 1].load(conv2, sess)
                                            conv_block_list.append(conv1)

                                        else:
                                            var_value, mean_value = computer_var_mean(pre_activation[id_layer+1])
                                            print(var_value)
                                            print(mean_value)
                                            conv_weight_1 = sess.run(var_con[id_layer+1])
                                            conv_block_list.append(conv_weight_1)
                                            mean_value_list.append(mean_value)

                                    conv_block_list_adjust=gutils.computer_between_layer(conv_block_list, mean_value_list)
                                    for j in range(start_layer, start_layer+block_depth):
                                        var_con[j].load(conv_block_list_adjust[j-start_layer], sess)

                                    # for k in range(block_depth - 1):
                                    #     conv_block_temp=sess.run(var_con[k])
                                    #     conv_block_list(temp)



                            values1 = self.full_validation([cross_entropy], sess, vali_data, vali_labels)
                            if values0[0] < values1[0]:
                                logging.info(' scaling is %s\n' % (values0[0] > values1[0]))
                                for k in range(totol_conv):
                                    var_con[k].load(conv_list[k], sess)
                            else:
                                for n in range(len(slots)):
                                    shape_M = slots[n].get_shape().as_list()
                                    slots[n].load(np.zeros(shape_M), sess)


                        elif (step > self.decay_step[0]) and (step <= self.decay_step[1]):
                            for k in range(1,5):
                                block_depth = depth[k]

                                start_layer = sum(depth[:k])
                                if block_depth > 1:
                                    conv_block_list = []
                                    mean_value_list = []
                                    for i in range(block_depth):

                                        if i < block_depth - 1:
                                            id_layer = start_layer + i
                                            var_value, mean_value = computer_var_mean(pre_activation[id_layer])
                                            mean_value_list.append(mean_value)
                                            print(var_value)
                                            print(mean_value)
                                            conv_weight_1, conv_weight_2 = sess.run(
                                                [var_con[id_layer], var_con[id_layer + 1]])
                                            scaling_value = var_value / mean_value
                                            print(scaling_value)
                                            conv1, conv2 = gutils.conv_scal_select(conv_weight_1, conv_weight_2,
                                                                                   scaling_value)
                                            var_con[id_layer].load(conv1, sess)
                                            var_con[id_layer + 1].load(conv2, sess)
                                            conv_block_list.append(conv1)

                                        else:
                                            var_value, mean_value = computer_var_mean(pre_activation[id_layer + 1])
                                            print(var_value)
                                            print(mean_value)
                                            conv_weight_1 = sess.run(var_con[id_layer + 1])
                                            conv_block_list.append(conv_weight_1)
                                            mean_value_list.append(mean_value)

                                    conv_block_list_adjust = gutils.computer_between_layer(conv_block_list,
                                                                                           mean_value_list)
                                    for j in range(start_layer, start_layer + block_depth):
                                        var_con[j].load(conv_block_list_adjust[j - start_layer], sess)
                            values1 = self.full_validation([cross_entropy], sess, vali_data, vali_labels)
                            if values0[0] < values1[0]:
                                logging.info(' scaling is %s\n' % (values0[0] > values1[0]))
                                for k in range(depth[0], totol_conv):
                                    var_con[k].load(conv_list[k], sess)
                            else:
                                for n in range(len(slots)):
                                    shape_M = slots[n].get_shape().as_list()
                                    slots[n].load(np.zeros(shape_M), sess)
                                    # layer_weight_grad = sess.run(tf.gradients(cross_entropy, var_con[:totol_conv]), feed_dict=feed_dict)
                                    # layer_weight = sess.run(var_con[:totol_conv])
                                     layer_weight_adjust = backward_scaling_adjust_grad_mean(layer_weight_grad, layer_weight)

                                    # for k in range(len(layer_weight_adjust)):
                                    #   var_con[k].load(layer_weight_adjust[k], sess)

                        # else:  # step > (self.decay_step[1]) and (step <= self.decay_step[2]):
                        #     for k in range(2,5):
                        #         block_depth = depth[k]
                        #
                        #         start_layer = sum(depth[:k])
                        #         if block_depth > 1:
                        #             conv_block_list = []
                        #             mean_value_list = []
                        #             for i in range(block_depth):
                        #
                        #                 if i < block_depth - 1:
                        #                     id_layer = start_layer + i
                        #                     var_value, mean_value = computer_var_mean(pre_activation[id_layer])
                        #                     mean_value_list.append(mean_value)
                        #                     print(var_value)
                        #                     print(mean_value)
                        #                     conv_weight_1, conv_weight_2 = sess.run(
                        #                         [var_con[id_layer], var_con[id_layer + 1]])
                        #                     scaling_value = var_value / mean_value
                        #                     print(scaling_value)
                        #                     conv1, conv2 = gutils.conv_scal_select(conv_weight_1, conv_weight_2,
                        #                                                            scaling_value)
                        #                     var_con[id_layer].load(conv1, sess)
                        #                     var_con[id_layer + 1].load(conv2, sess)
                        #                     conv_block_list.append(conv1)
                        #
                        #                 else:
                        #                     var_value, mean_value = computer_var_mean(pre_activation[id_layer + 1])
                        #                     print(var_value)
                        #                     print(mean_value)
                        #                     conv_weight_1 = sess.run(var_con[id_layer + 1])
                        #                     conv_block_list.append(conv_weight_1)
                        #                     mean_value_list.append(mean_value)
                        #
                        #             conv_block_list_adjust = gutils.computer_between_layer(conv_block_list,
                        #                                                                    mean_value_list)
                        #             for j in range(start_layer, start_layer + block_depth):
                        #                 var_con[j].load(conv_block_list_adjust[j - start_layer], sess)
                        #     values1 = self.full_validation([cross_entropy], sess, vali_data, vali_labels)
                        #     if values0[0] < values1[0]:
                        #         logging.info(' scaling is %s\n' % (values0[0] > values1[0]))
                        #         for k in range(depth[0] + depth[1], totol_conv):
                        #             var_con[k].load(conv_list[k], sess)


                    elif FLAGS.model == 'resnet':

                        for con in var_con:
                            con_unit = gutils.conv_adjust(con.eval(), False)
                            con.load(con_unit, sess)
                            # sess.run(tf.assign(con, con_unit))

                    elif FLAGS.model == 'resnet_sc':
                        for h in range(2, 5):
                            conv1_sc, conv2_sc = sess.run([var_con[h][0], var_con[h][1]])
                            conv1_sc, conv2_sc = gutils.conv_scal(conv1_sc, conv2_sc)
                            mean = [np.mean(conv1_sc), np.mean(conv2_sc)]
                            if (mean[0] * mean[1] >= 1e-10):
                                sca = np.sqrt(mean[0] * mean[1])
                                conv1_sc, conv2_sc = conv1_sc * (sca / mean[0]), conv2_sc * (sca / mean[1])
                                var_con[h][0].load(conv1_sc, sess)
                                var_con[h][1].load(conv2_sc, sess)
                                # sess.run([tf.assign(var_con[h][0], conv1_sc), tf.assign(var_con[h][1],conv2_sc)])

                    elif FLAGS.model == 'resnet_sc_bn':
                        for h in range(2, 5):
                            conv1_sc, conv2_sc = sess.run([var_con[h][0], var_con[h][1]])
                            conv1_sc, conv2_sc = gutils.conv_scal(conv1_sc, conv2_sc)
                            conv2_sc = gutils.conv_adjust(conv2_sc, False)
                            var_con[h][0].load(conv1_sc, sess)
                            var_con[h][1].load(conv2_sc, sess)
                            # sess.run([tf.assign(var_con[h][0], conv1_sc), tf.assign(var_con[h][1], conv2_sc)])

                    else:

                        assert False, 'unknown model'

                    # for n in range(len(slots)):
                    #     shape_M = slots[n].get_shape().as_list()
                    #     slots[n].load(np.zeros(shape_M), sess)

                        # if step % 1000==0 :
                        #   gra=sess.run(tf.gradients(losses, var_con), feed_dict=feed_dict)
                        #   for item in gra:
                        #     tempp1 = np.abs(item)
                        #     logging.info('gramaxvalue= %f graminvalue= %f grameanvalue= %f'%(np.max(tempp1), np.min(tempp1),np.mean(tempp1)))

            logging.info('total training time: %d sec' % (time.time() - t0))

        return


def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.train_path == None:
        t0 = time.gmtime()
        timestamp = '_%4d%02d%02d_%02d%02d%02d' % (t0.tm_year, t0.tm_mon, t0.tm_mday, t0.tm_hour, t0.tm_min, t0.tm_sec)
        train_path = os.path.join(FLAGS.train_dir, FLAGS.model + '_' + FLAGS.task + '_' + FLAGS.data + timestamp)
    else:
        train_path = FLAGS.train_path

    if not tf.gfile.Exists(train_path):
        print('creating directory %s' % train_path)
        tf.gfile.MakeDirs(train_path)

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(os.path.join(train_path, 'log.txt')), logging.StreamHandler()]
    # handlers = [logging.FileHandler(os.path.join('/home/robot/yuanqunyong')), logging.StreamHandler()]
    logging.basicConfig(level=level, format=format, handlers=handlers)

    logging.info(' '.join(sys.argv))

    trainer = Trainer()
    if FLAGS.task == 'train':
        trainer.train(train_path)
    elif FLAGS.task == 'test':
        trainer.test()
    elif FLAGS.task == 'see':
        trainer.see()
    elif FLAGS.task == 'train_scaling':
        trainer.train_scaling(train_path)
    else:
        assert False, 'unknown task'

    return


if __name__ == '__main__':
    tf.app.run()
