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
                 trainable=True):
        self.name = name
        self.kernel_size = [kernel_size[0],
                            kernel_size[1], in_depth, out_depth]
        self.trainable = trainable
        self.weights = tf.get_variable(name=self.name+'/weights',
            shape=self.kernel_size, dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=self.trainable)
        self.bias_weights = None
        self.values = None
        self.prune_mask = None
        self.strides = strides
        self.padding = padding
        self.centroids = []

        self.pruned_weights = tf.placeholder(
            tf.float32, self.weights.get_shape().as_list())
        self.assign_op = tf.assign(self.weights, self.pruned_weights)

        self.clusters_ph = tf.placeholder(tf.float32, self.weights.get_shape().as_list())
        self.assign_clusters_op = tf.assign(self.weights, self.clusters_ph)
        self.cast_op = tf.cast(self.weights, tf.int32)

    def assign_weights(self, init_weights):
        self.weights = tf.Variable(init_weights, dtype=tf.float32,
            name=self.name+'/weights', trainable=self.trainable)

    def assign_bias_weights(self, init_bias_weights):
        self.bias_weights = tf.Variable(init_bias_weights, dtype=tf.float32,
            name=self.name+'/biases', trainable=self.trainable)

    @abstractmethod
    def forward(self, input_tensor):
        self.values = None
        return self.values

    def refine_threshold(self, weights, sparsity_level):
        sorted_weights = np.sort(np.abs(weights).flatten())
        threshold_idx = int(sparsity_level * sorted_weights.size)
        threshold = sorted_weights[threshold_idx]
        return sorted_weights[threshold_idx]

    def prune_weights(self, session, pruning_threshold=None, sparsity_level=0.0):
        weight_values = np.asarray(session.run(self.weights))
        
        if pruning_threshold is None:
            pruning_threshold = self.refine_threshold(weight_values, sparsity_level)

        self.prune_mask = np.copy(weight_values)
        self.prune_mask[np.abs(self.prune_mask) > pruning_threshold] = 1
        self.prune_mask[self.prune_mask != 1] = 0
        print('set to 0', np.count_nonzero(self.prune_mask))
        # assign values
        session.run(self.assign_op, feed_dict={
                    self.pruned_weights: weight_values * self.prune_mask})

    def prune_gradients(self, grad):
        return grad * self.prune_mask

    def quantize_weights(self, session, num_clusters):
        weight_values = session.run(self.weights)

        # assign to clusters
        self.centroids = np.linspace(
            start=np.min(weight_values), 
            stop=np.max(weight_values), 
                num=num_clusters)

        self.weight_clusters = np.digitize(
            weight_values.flatten(), self.centroids)
        self.weight_clusters -= 1

        quantized_weight_values = [self.centroids[i] for i in self.weight_clusters]
        
        self.weight_clusters = np.reshape(
            self.weight_clusters, self.kernel_size)
        quantized_weight_values = np.reshape(
            quantized_weight_values, self.kernel_size)
        
        session.run(self.assign_clusters_op, feed_dict={
                    self.clusters_ph: quantized_weight_values})

        self.recompute_centroids(weight_values)


    def recompute_centroids(self, weight_values):
        for i in range(0, len(self.centroids)):
            weights_mask = np.zeros(self.weight_clusters.shape)
            weights_mask[self.weight_clusters == i] = 1

            count = np.count_nonzero(weights_mask)

            if (count != 0):
                weights_centroid_sum = np.sum(
                    weight_values * weights_mask)
                self.centroids[i] = weights_centroid_sum / count

    def quantize_gradients(self, gradient_values):
        gradients = np.zeros(gradient_values.shape)

        for i in range(0, len(self.centroids)):
            weights_mask = np.zeros(self.weight_clusters.shape)
            weights_mask[self.weight_clusters == i] = 1            
            gradient_sum = np.sum(gradient_values * weights_mask)
            gradients += weights_mask * gradient_sum
        return gradients

    def assign_clusters(self, session):
        session.run(self.assign_clusters_op, feed_dict={self.clusters_ph: self.weight_clusters})
        session.run(self.cast_op)

        session.run(self.assign_bias_clusters_op, feed_dict={self.bias_clusters_ph: self.bias_clusters})
        session.run(self.cast_bias_op)

        return self.centroids