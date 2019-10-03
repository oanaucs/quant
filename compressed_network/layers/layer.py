from abc import ABC, abstractmethod

import tensorflow as tf

import numpy as np


class layer_base(ABC):
    def __init__(self,
                 in_depth,
                 out_depth,
                 kernel_size=[3, 3],
                 init_weights=None,
                 init_bias_weights=None,
                 padding='SAME',
                 name=None,
                 reuse=None,
                 num_clusters=8,
                 pruning_threshold=0.0015):
        self.name = name
        self.kernel_size = [kernel_size[0],
                            kernel_size[1], in_depth, out_depth]
        self.weights = tf.Variable(tf.random_normal(
            shape=self.kernel_size, stddev=0.1), 
            name=self.name+'/weights', trainable=True)
        self.bias_weights = None
        self.values = None
        self.prune_mask = None
        self.pruning_threshold = pruning_threshold
        self.strides = strides
        self.padding = padding
        self.num_clusters = num_clusters
        self.centroids = []
        self.pruned_weights = tf.placeholder(
            tf.float32, self.weights.get_shape().as_list())
        self.assign_op = tf.assign(self.weights, self.pruned_weights)

    def assign_weights(self, init_weights):
        self.weights = tf.Variable(init_weights, dtype=tf.float32,
            name=self.name+'/weights', trainable=True)

    def assign_bias_weights(self, init_bias_weights):
        self.bias_weights = tf.Variable(init_bias_weights, dtype=tf.float32,
            name=self.name+'/biases', trainable=True)
        # print('assigned bias', self.name, self.bias_weights.shape)


    @abstractmethod
    def forward(self, input_tensor):
        self.values = None
        return self.values

    def prune_weights(self, session):
        weight_values = np.asarray(session.run(self.weights))
        self.prune_mask = np.copy(weight_values)
        self.prune_mask[self.prune_mask < self.pruning_threshold] = 0
        self.prune_mask[self.prune_mask != 0] = 1
        session.run(self.assign_op, feed_dict={
                    self.pruned_weights: weight_values * self.prune_mask})

    def prune_gradients(self, grad):
        return grad * self.prune_mask

    def quantize_weights(self, session):
        weight_values = session.run(self.weights)
        # assign to clusters
        self.centroids = np.linspace(start=np.min(weight_values), stop=np.max(
            weight_values), num=self.num_clusters)

        self.previous_centroids = np.copy(self.centroids)
        clustered_weights = np.digitize(
            weight_values.flatten(), self.centroids)

        quantized_values = [self.centroids[i-1] for i in clustered_weights]
        quantized_values = np.reshape(
            quantized_values, self.kernel_size)

        # recompute centroids
        for i in range(0, len(self.centroids)):
            centroid_mask = np.copy(quantized_values)
            centroid_mask[centroid_mask != self.centroids[i]] = 0
            centroid_mask[centroid_mask == self.centroids[i]] = 1
            centroid_count = np.count_nonzero(centroid_mask)

            if (centroid_count != 0):
                self.centroids[i] = np.sum(
                    weight_values * centroid_mask) / centroid_count
        session.run(self.assign_op, feed_dict={
                    self.pruned_weights: quantized_values})

    def quantize_gradients(self, gradient_values):
        clustered_gradients = np.digitize(
            gradient_values.flatten(), self.centroids)
        clustered_gradients = np.reshape(clustered_gradients, self.kernel_size)

        gradients = np.zeros(gradient_values.shape)
        for i in range(0, len(self.centroids)):
            centroid_mask = np.copy(clustered_gradients)
            centroid_mask[centroid_mask != self.centroids[i]] = 0
            centroid_mask[centroid_mask == self.centroids[i]] = 1
            gradient_sum = np.sum(gradient_values * centroid_mask)
            gradients += centroid_mask * gradient_sum
        return gradients