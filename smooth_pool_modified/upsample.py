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
from preprocess import load_batch 

from layers import act
from layers import smooth_layer
from layers import densepool_layer



def inference(input, batch_size, tri_num, vert_num, mtx, mtx_1, phase, keep_prob):
    """Build the Upsamle model.
    Args:
      input: 
      dim, mtx, mtx_1:
    Returns:
      output.
    """
    print("=============== Inference: Build the model =============== ")
    print("input.shape:   ", input.shape)       #(?, 1292, 9)
    # input_tensor = tf.convert_to_tensor(tf.transpose(input, perm=[1, 0, 2]))    #(1292, batch_size, 9)
    # input_tensor32 = tf.cast(input_tensor, tf.float32)

    ## add noise
    alpha = 0.1
    noise_tensor = tf.random_normal([1292, 9], seed=1234)   #(1292, 9)
    input_add_noise = input + alpha * noise_tensor
    
    input_tensor = tf.convert_to_tensor(tf.transpose(input_add_noise, perm=[1, 0, 2]))    #(1292, batch_size, 9)
    input_tensor32 = tf.cast(input_tensor, tf.float32)

    # tf.summary.histogram('input_tensor', input_tensor32)

    mtx_tensor = tf.convert_to_tensor(mtx, dtype='float32', name='mtx_tensor')
    mtx_1_tensor = tf.transpose(tf.convert_to_tensor(mtx_1, dtype='float32', name='mtx_1_tensor'))

    smooth1 = smooth_layer(input_tensor32, tri_num, "smooth1")
    pool1 = densepool_layer(smooth1, tri_num, "pool1", mtx_tensor, mtx_1_tensor)
    # block1 = tf.contrib.layers.batch_norm(pool1, center=True, scale=True, is_training=phase, scope='bn') 

    # output_tensor = tf.transpose(pool1, perm=[1, 0, 2])

    drop_out = tf.nn.dropout(pool1, keep_prob)
    output_tensor = tf.transpose(drop_out, perm=[1, 0, 2])
    
    # return act(output_tensor)
    return output_tensor


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
    opt = tf.train.AdamOptimizer(lr)

    # Compute the gradients for a list of variables.
    grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
    # print(grads_and_vars)       # (1292, 9, 9)

    # test #
    # graph_coll = tf.Graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # print(graph_coll)
    # print("grads_and_vars: ", grads_and_vars)   # it's a tuple <tensor, variable>

    ########### debug ###############################################
    W = [v for v in tf.trainable_variables() if v.name == "smooth1/weights/Variable:0"][0]
    grad_W = tf.gradients(xs=[W], ys=loss)
    gw = grad_W / tf.norm(grad_W)
    # # gw = tf.reduce_mean(grad_W)
    # # print("grad_W: ", grad_W)    # tensor (700, 187)
    tf.summary.scalar('grad_W', tf.reduce_mean(grad_W))
    tf.summary.scalar('gw', tf.reduce_mean(gw))

    # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
    # need to the 'gradient' part, for example cap them, etc.
    capped_grads_and_vars = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in grads_and_vars]

    # Ask the optimizer to apply the capped gradients.
    train_op = opt.apply_gradients(capped_grads_and_vars, global_step)
    # train_op = opt.apply_gradients(grads_and_vars, global_step)

    # train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step)
    
    # train_op = tf.train.AdagradOptimizer(lr).minimize(loss, global_step)
    # momentum = 0.9
    # train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step)
  
  return train_op, lr


##
## @brief      { loss function }
##
## @param      ground_truth  The ground truth
## @param      predictions   The predictions
##
## @return     { description_of_the_return_value }
##
def loss(ground_truth, predictions):
  print("loss >>> ground_truth: ", ground_truth)
  print("         predictions: ", predictions)  # Tensor("Shape:0", shape=(3,), dtype=int32)

  with tf.name_scope("loss"):
    weights = [v for v in tf.trainable_variables() if v.name == "smooth1/weights/Variable:0"][0]
    regularizer = tf.nn.l2_loss(weights)

    loss_mse = tf.losses.mean_squared_error(ground_truth, predictions)
    # loss_mean = tf.reduce_mean(tf.squared_difference(ground_truth, predictions))
    
    beta = 0.05
    loss = tf.reduce_mean(loss_mse + beta * regularizer)
  
  tf.summary.scalar('loss_mse', loss_mse)
  tf.summary.histogram('ground_truth', ground_truth)
  tf.summary.histogram('predictions', predictions)

  return loss_mse


