# layers.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys

import tensorflow as tf
import numpy as np
import util

def weight_variable(shape):
	"""Create a weight variable with appropriate initialization."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	"""Create a bias variable with appropriate initialization."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
			tf.summary.scalar('sttdev/' + name, stddev)
			tf.summary.scalar('max/' + name, tf.reduce_max(var))
			tf.summary.scalar('min/' + name, tf.reduce_min(var))
			tf.summary.histogram(name, var)


# input_tensor: (1292, batch_size, 9)
def smooth_layer(input_tensor, tri_num, layer_name, act=tf.nn.relu):
	""" smooth layer, learn weights for each triangle"""
	with tf.name_scope(layer_name):
		with tf.name_scope("weights"):
			weights = weight_variable([tri_num, 9, 9])
			variable_summaries(weights, "weights")
		# with tf.name_scope("biases"):
		# 	biases = bias_variable([9])
		# 	variable_summaries(biases, "smooth-biases")
		with tf.name_scope('XW_plus_b'):
			smooth_out = tf.matmul(input_tensor, weights)  #(1292, 20, 9) x (1292, 9, 9)
			tf.summary.histogram("smooth_out", smooth_out)
		if act is not None:
			activations = act(preactivate, name='activation')
			tf.summary.histogram('activations', activations)
			return activations
		else:
			return smooth_out


# input_tensor: (1292, batch_size, 9)
def densepool_layer(input_tensor, tri_num, layer_name, mtx_tensor, mtx_1_tensor, act=tf.nn.relu):
	""" dense pool layer, don't have weights, don't learn anything """
	with tf.name_scope(layer_name):
		permute_input = tf.transpose(input_tensor, perm=[1, 0, 2])   # (b, 1292, 9)
		flat_input = tf.reshape(input_tensor, [-1, tri_num * 3, 3])     # (b, 1292*3, 3) 

		# pos = tf.matmul(tf.expand_dims(mtx_tensor, 0), flat_input)        # (b, 700, 3)
		pos = tf.einsum('jk,ikl->ijl', mtx_tensor, flat_input)
		print("pos:  ", pos)
		# new_pos = tf.transpose(pos, perm=[1, 0, 2])    

		# tri = tf.matmul(tf.expand_dims(mtx_1_tensor, 0), new_pos)  # (1292*3, 700) * (b, 700, 3) = (b, 1292*3, 3)
		tri = tf.einsum('jk,ikl->ijl', mtx_1_tensor, pos)
		print("tri:  ", tri)
		# new_pos = tf.transpose(tri, perm=[1, 0, 2])

		preactivate = tf.reshape(tri, [-1, tri_num, 9])

		if act is not None:
			activations = act(preactivate, name="activation")
			return activations
		else:
			return preactivate



# test: full connected layer
# size_in:  M
# size_out: N
# weights: N * M
def fc_layer(input, batch_size, size_in, size_out, name="fc"):
  print("batch_size::m::n:: ", batch_size, size_in, size_out)
  print("input: ", input)

  # batch_size::m::n::  20 187 700
  # input:  Tensor("Reshape:0", shape=(187, ?), dtype=float32)

  with tf.name_scope(name):
    W = tf.Variable(tf.random_normal([size_out, size_in], mean=-0.001, stddev=0.05), name="W")
    y = tf.matmul(W, input) 
    # tf.summary.histogram("weights", W)
    return y


