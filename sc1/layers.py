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



# activation function
def act(x):
	return tf.abs(tf.sigmoid(x) - 0.5)


def weight_variable(shape):
	"""Create a weight variable with appropriate initialization."""
	initial = tf.truncated_normal(shape, mean=-0.001, stddev=0.01)
	# initial = tf.constant(0.3, shape=shape)
	return tf.Variable(initial, name="weights")

def bias_variable(shape):
	"""Create a bias variable with appropriate initialization."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


# def variable_summaries(var, name):
# 	"""Attach a lot of summaries to a Tensor."""
# 	with tf.name_scope('summaries'):
# 		mean = tf.reduce_mean(var)
# 		tf.summary.scalar('mean/' + name, mean)
		# with tf.name_scope('stddev'):
		# 	stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
			# tf.summary.scalar('sttdev/' + name, stddev)
			# tf.summary.scalar('max/' + name, tf.reduce_max(var))
			# tf.summary.scalar('min/' + name, tf.reduce_min(var))
			# tf.summary.histogram(name, var)


# input_tensor: (1292, batch_size, 9)
def smooth_layer(input_tensor, tri_num, layer_name):
	""" smooth layer, learn weights for each triangle"""
	with tf.name_scope(layer_name):
		with tf.name_scope("weights"):
			weights = weight_variable([tri_num, 9, 9])
			tf.summary.histogram("weights", weights)
		# with tf.name_scope("biases"):
		# 	biases = bias_variable([9])
		# 	variable_summaries(biases, "smooth-biases")
		with tf.name_scope('XW'):
			smooth_out = tf.matmul(input_tensor, weights)  #(1292, 20, 9) x (1292, 9, 9)
		
		return smooth_out


# input_tensor: (1292, batch_size, 9)
def densepool_layer(input_tensor, tri_num, layer_name, mtx_tensor, mtx_1_tensor):
	""" dense pool layer, don't have weights, don't learn anything """
	with tf.name_scope(layer_name):
		permute_input = tf.transpose(input_tensor, perm=[1, 0, 2])   # (b, 1292, 9)
		flat_input = tf.reshape(input_tensor, [-1, tri_num * 3, 3])     # (b, 1292*3, 3) 

		# pos = tf.matmul(tf.expand_dims(mtx_tensor, 0), flat_input)        # (b, 700, 3)
		pos = tf.einsum('jk,ikl->ijl', mtx_tensor, flat_input)
		# print("pos:  ", pos)

		# tri = tf.matmul(tf.expand_dims(mtx_1_tensor, 0), new_pos)  # (1292*3, 700) * (b, 700, 3) = (b, 1292*3, 3)
		tri = tf.einsum('jk,ikl->ijl', mtx_1_tensor, pos)
		# print("tri:  ", tri)

		tri_reshape = tf.reshape(tri, [-1, tri_num, 9])
		output = tf.transpose(tri_reshape, perm=[1, 0, 2])

		return output



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
    W = tf.Variable(tf.random_normal([size_out, size_in], mean=-0.001, stddev=0.05), name="fc_W")
    y = tf.matmul(input, W) 
    return y


# element wise multiply 
def elem_fc_layer(input, num_v, name="elem_fc"):
	with tf.name_scope(name):
		W = tf.Variable(tf.random_normal([num_v, 1], mean=-0.001, stddev=0.05), name="weights")
		y = input * W
		return y


##
## @brief      { ? x 700 x 3 }
##
## @param      input_tensor  The input tensor
## @param      mask_tensor   The mask tensor
## @param      num_v         vertex total
## @param      layer_name    The layer name
##
## @return     { ? x 700 x 3 }
##
def sparsely_connected(input_tensor, mask_tensor, num_v, layer_name):
	"""
		Masking .(element wise) Weights * X
	"""
	with tf.name_scope(layer_name):
		weights = weight_variable([num_v, num_v])
		tf.summary.histogram("weights", weights)
		sparsely_w = tf.multiply(mask_tensor, weights)             
			# sparsely_w = tf.expand_dims(tf.multiply(mask_tensor, weights), 0)  
			# x = tf.transpose(input_tensor, perm=[0, 2, 1])            

			# sparsely_out = tf.matmul(sparsely_w, input_tensor)  #  (1, 700, 700) x (batch, 700, 3) 
			# sparsely_out = tf.matmul(x, sparsely_w)  #  (1, 700, 700)   (batch, 700, 3) 
			# sparsely_out = tf.tensordot(input_tensor, sparsely_w, [[1], [0]]) #???
			# print(sparsely_out)
			# 
		input_tensor_trans = tf.transpose(input_tensor, perm=[0, 2, 1])  #(batch, 3, 700)
		_Y = tf.tensordot(input_tensor_trans, sparsely_w, [[2], [0]])
		print("_Y:  ", _Y)

		sp_out = tf.transpose(_Y, perm=[0, 2, 1])  
		print("sp_out:  ", sp_out)
		
		return sp_out