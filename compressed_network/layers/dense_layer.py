from layers.layer import layer_base

import tensorflow as tf

import numpy as np


class dense(layer_base):
    def __init__(self,
                 input_shape,
                 out_depth,
                 name=None,
                 reuse=None,
                 num_clusters=64,
                 pruning_threshold=-0.05):
        self.name = name
        self.kernel_size = [input_shape[1], out_depth]
        self.weights = tf.get_variable(name=self.name+'/weights',
            shape=self.kernel_size, dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        self.bias_weights = None
        self.values = None
        self.prune_mask = None
        self.pruning_threshold = pruning_threshold
        self.num_clusters = num_clusters
        self.centroids = []
        
        self.pruned_weights = tf.placeholder(tf.float32, self.weights.get_shape().as_list())
        self.assign_op = tf.assign(self.weights, self.pruned_weights)

        self.clusters_ph = tf.placeholder(tf.float32, self.weights.get_shape().as_list())
        self.assign_clusters_op = tf.assign(self.weights, self.clusters_ph)
        self.cast_op = tf.cast(self.weights, tf.int32)

    def forward(self, input_tensor):
        self.values = tf.matmul(input_tensor, self.weights)

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