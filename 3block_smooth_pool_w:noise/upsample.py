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
from layers import smooth_layer
from layers import densepool_layer



def inference(input, batch_size, tri_num, vert_num, mtx, mtx_1, phase):
    """Build the Upsamle model.
    Args:
      input: 
      dim, mtx, mtx_1:
    Returns:
      output.
    """
    print("=============== Inference: Build the model =============== ")
    # print("input.shape:   ", input.shape)       #(?, 1292, 9)

    # add noise
    noise_tensor = tf.random_normal([1292, 9], seed=1234)   #(1292, 9)
    input_add_noise = input + 0.001 * noise_tensor
    
    input_tensor = tf.convert_to_tensor(tf.transpose(input_add_noise, perm=[1, 0, 2]))    #(1292, batch_size, 9)
    # input_tensor = tf.convert_to_tensor(tf.transpose(input, perm=[1, 0, 2]))    #(1292, batch_size, 9)
    input_tensor32 = tf.cast(input_tensor, tf.float32)

    # input_flat = tf.reshape(input_tensor32, [size_in, batch_size * int(input.shape[2])])    #(1292, -1)
    # input_dim = output_dim = tf.shape(input_flat)
    tf.summary.histogram('input_tensor', input_tensor32)

    # Note: in keras
    # mtx_tensor = K.constant(self.mtx, dtype='float32', name='mtx_tensor')   # v_num x (tri*3)
    # mtx_1_tensor = K.transpose(K.constant(self.mtx_1, dtype='float32', name='mtx_1_tensor')) 
    mtx_tensor = tf.convert_to_tensor(mtx, dtype='float32', name='mtx_tensor')
    mtx_1_tensor = tf.transpose(tf.convert_to_tensor(mtx_1, dtype='float32', name='mtx_1_tensor'))

    # (1292, batch_size, 9)
    print("input_tensor32: ", input_tensor32)
    smooth1 = smooth_layer(input_tensor32, tri_num, "smooth1", None)
    pool1 = densepool_layer(smooth1, tri_num, "pool1", mtx_tensor, mtx_1_tensor, None)
    print(">>> pool1: ", pool1)

    b1 = tf.transpose(pool1, perm=[1, 0, 2])
    bn1 = tf.contrib.layers.batch_norm(b1, center=True, scale=True, is_training=phase, scope='bn1') 
    block1 = tf.transpose(bn1, perm=[1, 0, 2])
    print(">>> block1: ", block1)

    smooth2 = smooth_layer(block1, tri_num, "smooth2", None)
    pool2 = densepool_layer(smooth2, tri_num, "pool2", mtx_tensor, mtx_1_tensor, None)
    print(">>> pool2: ", pool2)

    b2 = tf.transpose(pool2, perm=[1, 0, 2])
    bn2 = tf.contrib.layers.batch_norm(b2, center=True, scale=True, is_training=phase, scope='bn2') 
    block2 = tf.transpose(bn2, perm=[1, 0, 2])
    print(">>> block2: ", block2)

    smooth3 = smooth_layer(block2, tri_num, "smooth1", None)
    pool3 = densepool_layer(smooth3, tri_num, "pool1", mtx_tensor, mtx_1_tensor, None)
    print(">>> pool3: ", pool3)

    output_tensor = tf.transpose(pool3, perm=[1, 0, 2])

    # return tf.nn.relu(block1, 'relu')
    return output_tensor


def training(learning_rate, loss, global_step, lr_decay_rate):
  # Decay the learning rate exponentially based on the number of steps.
  if lr_decay_rate is not 0:
    lr = tf.train.exponential_decay(learning_rate,
                                    global_step,
                                    500, lr_decay_rate,
                                    staircase=True)
  else:
    lr = tf.convert_to_tensor(learning_rate, dtype=tf.float32, name="learning_rate")

  # boundaries = [5000, 10000, 50000, 100000]
  # values = [0.1, 0.05, 0.01, 0.005, 0.001]
  # lr = tf.train.piecewise_constant(global_step, boundaries, values)

  tf.summary.scalar('learning_rate', lr)
  
  with tf.name_scope("train"):
    # Create the gradient descent optimizer with the given learning rate.
    opt = tf.train.GradientDescentOptimizer(lr)

    # Compute the gradients for a list of variables.
    grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
    # print(grads_and_vars)       # (1292, 9, 9)

    # test #
    # graph_coll = tf.Graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # print(graph_coll)
    # print("grads_and_vars: ", grads_and_vars)   # it's a tuple <tensor, variable>

    ########### debug ###############################################
    # W = [v for v in tf.trainable_variables() if v.name == "smooth1/weights:0"][0]  
    # grad_W = tf.gradients(xs=[W], ys=loss)
    # gw = grad_W / tf.norm(grad_W)
    # # gw = tf.reduce_mean(grad_W)
    # # print("grad_W: ", grad_W)    # tensor (700, 187)
    # tf.summary.scalar('grad_W', tf.reduce_mean(grad_W))
    # tf.summary.scalar('gw', tf.reduce_sum(gw))

    # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
    # need to the 'gradient' part, for example cap them, etc.
    # capped_grads_and_vars = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in grads_and_vars]

    # Ask the optimizer to apply the capped gradients.
    # train_op = opt.apply_gradients(capped_grads_and_vars, global_step)
    # train_op = opt.apply_gradients(grads_and_vars, global_step)

    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  
  return train_op, lr


def gamma(N, a, b, c):
  gam = np.zeros((N, N))
  for i in range(0, N):
    t = 1 + b * i/N;
    g = a * pow(t, c);
    # gam[i][i] = 1.0 / ( 0.2 + g);   
    gam[i][i] = 0.2 + g;   
  return gam
    

def loss(ground_truth, predictions):
  print("loss >>> ground_truth: ", ground_truth)
  print("         predictions: ", predictions)   

  with tf.name_scope("loss"):
    # loss = tf.losses.mean_squared_error(ground_truth, predictions, 
    #   weights=1.0,
    #   scope=None,
    #   loss_collection=tf.GraphKeys.LOSSES,
    #   reduction=tf.losses.Reduction.MEAN)
    loss_mse = tf.losses.mean_squared_error(ground_truth, predictions)
    # loss_mean = tf.reduce_mean(tf.squared_difference(ground_truth, predictions))
    
    # loss += tf.reduce_mean(0.01*tf.nn.l2_loss(u_weights))
  
  # tf.summary.scalar('loss_mse', loss_mse)
  tf.summary.histogram('ground_truth', ground_truth)
  tf.summary.histogram('predictions', predictions)

  return loss_mse


