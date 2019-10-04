import tensorflow as tf

import numpy as np

from layers.conv_layer import conv2d
from layers.dense_layer import dense


class CNN():
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.create_network_model()

    def create_network_model(self):
        self.c1 = conv2d(in_depth=1, out_depth=32, kernel_size=[5,5], name='conv1')
        self.c2 = conv2d(in_depth=32, out_depth=64, kernel_size=[5,5], name='conv2')
        self.fc3 = dense(input_shape=[self.batch_size, 7* 7 * 64], 
            out_depth=1024, name='fc3')
        self.logits = dense(input_shape=[self.batch_size, 1024], 
            out_depth=self.num_classes, name='logits')


    def forward_pass(self, images):
        c1_values = self.c1.forward(images)
        maxpool1 = tf.contrib.layers.max_pool2d(c1_values, [2, 2], scope='pool1')

        c2_values = self.c2.forward(maxpool1)
        maxpool2 = tf.contrib.layers.max_pool2d(c2_values, [2, 2], scope='pool2')

        flat_maxpool2 = tf.reshape(maxpool2, [self.batch_size, 7 * 7* 64]) 
        fc3_values = self.fc3.forward(flat_maxpool2)

        logits_values = self.logits.forward(fc3_values)
        predictions = tf.cast(tf.argmax(input=logits_values, axis=1), tf.int32)

        return logits_values, predictions

    def backward_pass(self):
        pass

    def train(self, images):
        return self.forward_pass(images)

    def layers_to_compress(self):
        return [self.c1, self.c2, self.fc3]