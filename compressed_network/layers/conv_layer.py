from layers.layer import layer_base

import tensorflow as tf

import numpy as np


class conv2d(layer_base):
    def __init__(self,
                 in_depth,
                 out_depth,
                 kernel_size=[3, 3],
                 strides=[1, 1, 1, 1],
                 padding='SAME',
                 name=None,
                 reuse=None):
        self.name = name
        self.kernel_size = [kernel_size[0],
                            kernel_size[1], in_depth, out_depth]
        self.weights = tf.get_variable(name=self.name+'/weights',
            shape=self.kernel_size, dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        self.bias_weights = None
        self.values = None
        self.strides = strides
        self.padding = padding
        
        self.prune_mask = None

        self.centroids = []
        self.name = name
        
        self.pruned_weights = tf.placeholder(tf.float32, self.weights.get_shape().as_list())
        self.assign_op = tf.assign(self.weights, self.pruned_weights)

        self.clusters_ph = tf.placeholder(tf.float32, self.weights.get_shape().as_list())
        self.assign_clusters_op = tf.assign(self.weights, self.clusters_ph)
        self.cast_op = tf.cast(self.weights, tf.int32)

    def forward(self, input_tensor):
        self.values = tf.nn.conv2d(
            input_tensor, self.weights, strides=self.strides, padding=self.padding,
            name=self.name)

        if self.bias_weights is None:
            values_shape = self.values.get_shape().as_list()
            bias_shape = [values_shape[-1]]
            
            self.bias_weights = tf.get_variable(name=self.name+'/biases',
                shape=bias_shape, dtype=tf.float32, 
                initializer=tf.zeros_initializer(),
                trainable=True)

        self.bias_values = tf.nn.bias_add(self.values, self.bias_weights)
        self.relu_values = tf.nn.relu(self.bias_values)

        return self.relu_values

    def shape(self):
        if self.values:
            return self.values.shape()
        else:
            return None
