"""Builds the Upsample network.
Summary of available functions:
 # Convert input numpy array to tensor for training. 
 x_train, y_train = inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(y_train, predictions)

 # Optimizer with respect to the loss.
 optimizer = optimizer(loss)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys

import numpy as np
import tensorflow as tf
import util
from preprocess import load_sample

from layers import act
from layers import smooth_layer, densepool_layer
from layers import sparsely_connected, elem_fc_layer


# alpha: noise weight
# mtx, mtx_1,
def inference(x_train, batch_size, vert_num, mask, phase, keep_prob):
    """Build the Upsamle model.
    Args:
      input: 
      dim, mtx, mtx_1, mask:
    Returns:
      output.
    """
    print("=============== Inference: Build the model =============== ")
    print("input.shape:   ", x_train.shape)       #(?, 700, 3)
    # input_tensor = tf.convert_to_tensor(tf.transpose(input, perm=[1, 0, 2]))    #(1292, batch_size, 9)
    # input_tensor32 = tf.cast(input_tensor, tf.float32)

    xpos_tensor32 = tf.cast(x_train[:, :, 0:3], tf.float32)  # (?, 700, 3)
    xvel_tensor32 = tf.cast(x_train[:, :, 3:6], tf.float32)  # (?, 700, 3)
    # print(xvel_tensor32.shape)       # ==> TensorShape([Dimension(2), Dimension(3)])
    ##
    ## { position branch }
    ##
    mask_tensor = tf.convert_to_tensor(mask, dtype='float32', name='mask_tensor')
    sc1 = sparsely_connected(xpos_tensor32, mask_tensor, vert_num, "sc1")


    ##
    ## { velocity branch }
    ##
    dim = xvel_tensor32.shape.as_list()
    vel_addon = elem_fc_layer(xvel_tensor32, dim[1], name="elem_fc")  # (100, 700, 3)

    # return sc1
    out = sc1 + 0.1 * vel_addon
    return out


def training(learning_rate, loss, global_step, lr_decay_rate):
  # Decay the learning rate exponentially based on the number of steps.
  if lr_decay_rate is not 0:
    lr = tf.train.exponential_decay(learning_rate,
                                    global_step,
                                    500, lr_decay_rate,
                                    staircase=True,
                                    name="learning_rate_decay")  
  else:
    lr = tf.convert_to_tensor(learning_rate, dtype=tf.float32, name="learning_rate")
  
  # boundaries = [5000, 10000, 50000, 100000]
  # values = [0.1, 0.05, 0.01, 0.005, 0.001]
  # lr = tf.train.piecewise_constant(global_step, boundaries, values)

  tf.summary.scalar('learning_rate_decay', lr)
  
  with tf.name_scope("train"):
    
    # Create optimizer with the given learning rate.
    # opt = tf.train.AdamOptimizer(lr)

    # Compute the gradients for a list of variables.
    # grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
    # print(grads_and_vars)       # (1292, 9, 9)

    # test #
    # graph_coll = tf.Graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # print(graph_coll)
    # print("grads_and_vars: ", grads_and_vars)   # it's a tuple <tensor, variable>

    ########### debug ###############################################
    # W = [v for v in tf.trainable_variables() if v.name == "sc1/weights/Variable:0"][0]
    # grad_W0 = tf.gradients(xs=W, ys=loss)[0]
    # print("grad_W0: ", grad_W0)    # tensor (700, 187)
    
    # grad_W = [grad for grad, var in grads_and_vars][0]
    # print("grad_W: ", grad_W) 

    # grad_w_diff = grad_W0 - grad_W
    # tf.summary.scalar('grad_w_diff', tf.reduce_mean(grad_w_diff))

    # gw = grad_W / W
    # tf.summary.scalar('gw_mean', tf.reduce_mean(gw))

    # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
    # need to the 'gradient' part, for example cap them, etc.
    # capped_grads_and_vars = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in grads_and_vars]

    # Ask the optimizer to apply the capped gradients.
    # train_op = opt.apply_gradients(capped_grads_and_vars, global_step)
    # train_op = opt.apply_gradients(grads_and_vars, global_step)

    opt = tf.train.AdamOptimizer(lr).minimize(loss, global_step)
    
    # train_op = tf.train.AdagradOptimizer(lr).minimize(loss, global_step)
    # momentum = 0.9
    # train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step)
  
  return opt, lr


##
## @brief      { loss function }
##
## @param      ground_truth  The ground truth
## @param      predictions   The predictions
## @param      beta          L2 weight
##
## @return     { description_of_the_return_value }
##
def loss(ground_truth, predictions, mask, alpha, beta):
  print("loss >>> ground_truth: ", ground_truth)
  print("         predictions: ", predictions)  # Tensor("Shape:0", shape=(3,), dtype=int32)

  suffix = '_' + str(alpha) + '_' + str(beta)
  with tf.name_scope("loss" + suffix):
    # weights = [v for v in tf.trainable_variables() if v.name == "smooth1/weights/Variable:0"][0]
    # weights = [v for v in tf.trainable_variables() if v.name == "sc1/weights/Variable:0"][0]
    # regularizer = tf.nn.l2_loss(weights)
    # print(tf.trainable_variables())

    sc1_w = [v for v in tf.trainable_variables() if v.name == "sc1/weights:0"]
    elem_fc_w = [v for v in tf.trainable_variables() if v.name == "elem_fc/weights:0"]

    print(" >>>> mask: >>>>")
    print(mask)
    mask_tensor = tf.convert_to_tensor(mask, dtype='float32', name='mask_tensor')
    m_sc1_w = tf.multiply(mask_tensor, sc1_w) 
    reg1 = alpha * tf.reduce_mean(tf.abs(ground_truth))

    
    # l1_regularizer_1 = tf.contrib.layers.l1_regularizer(scale=alpha, scope="regularizer")
    # l1_regularizer_2 = tf.contrib.layers.l1_regularizer(scale=beta, scope="regularizer")

    # regularizer1 = tf.contrib.layers.apply_regularization(l1_regularizer_1, m_sc1_w)
    # regularizer2 = tf.contrib.layers.apply_regularization(l1_regularizer_2, elem_fc_w)

    loss_mse = tf.losses.mean_squared_error(ground_truth, predictions)
    # loss_mean = tf.reduce_mean(tf.squared_difference(ground_truth, predictions))
    # loss_max = tf.reduce_max(tf.squared_difference(ground_truth, predictions))
    # 
    loss_mean = tf.reduce_mean(tf.square(ground_truth))
    
    loss = loss_mse + reg1
    # loss = loss_mse + regularizer1 + regularizer2
  
    tf.summary.scalar('loss', loss)
    # tf.summary.scalar('loss_max', loss_max)
    tf.summary.scalar('loss_mse', loss_mse)
    tf.summary.scalar('reg1', reg1)
    # tf.summary.scalar('regularizer1', regularizer1)
    # tf.summary.scalar('regularizer2', regularizer2)
    

    tf.summary.scalar('sc1_w' + suffix, tf.reduce_sum(tf.square(m_sc1_w)))
    tf.summary.scalar('elem_fc_w', tf.reduce_sum(tf.square(elem_fc_w)))

    tf.summary.histogram('ground_truth', ground_truth)
    tf.summary.histogram('predictions', predictions)

  return loss


