import tensorflow as tf

import numpy as np

from layers.conv_layer import conv2d
from layers.dense_layer import dense


class FCNN():
    def __init__(self, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.create_network_model()

    def create_network_model(self):
        self.c1 = conv2d(in_depth=1, out_depth=32, kernel_size=[3,3], name='conv1', padding='SAME')
        self.c2 = conv2d(in_depth=32, out_depth=32, kernel_size=[3,3], name='conv2', padding='SAME')
        self.c3 = conv2d(in_depth=32, out_depth=64, kernel_size=[3,3], name='conv3', padding='SAME')
        self.c4 = conv2d(in_depth=64, out_depth=128, kernel_size=[3,3], name='conv4', padding='SAME')

        self.fc5 = dense(input_shape=[self.batch_size, 3* 3 * 128], 
            out_depth=512, name='fc5')
        self.fc6 = dense(input_shape=[self.batch_size, 512], 
            out_depth=128, name='fc6')
        self.logits = dense(input_shape=[self.batch_size, 128], 
            out_depth=self.num_classes, name='logits')


    def forward_pass(self, images):
        c1_values = self.c1.forward(images)
        print('conv 1 shape', c1_values.shape)

        c2_values = self.c2.forward(c1_values)
        maxpool1 = tf.contrib.layers.max_pool2d(c1_values, [2, 2], scope='pool1')

        c3_values = self.c3.forward(maxpool1)
        maxpool2 = tf.contrib.layers.max_pool2d(c3_values, [2, 2], scope='pool2')

        c4_values = self.c4.forward(maxpool2)
        maxpool3 = tf.contrib.layers.max_pool2d(c4_values, [2, 2], scope='pool3')

        print('conv 4 shape', maxpool3.shape)

        flat_maxpool3 = tf.reshape(maxpool3, [self.batch_size, 3 * 3* 128]) 
        fc5_values = self.fc5.forward(flat_maxpool3)

        fc6_values = self.fc6.forward(fc5_values)

        logits_values = self.logits.forward(fc6_values)

        logits_values = tf.squeeze(logits_values, name='logits/squeeze')
        predictions = tf.cast(tf.argmax(input=logits_values, axis=1), tf.int32)

        return logits_values, predictions

    def backward_pass(self):
        pass

    def train(self, images):
        return self.forward_pass(images)

    def layers_as_list(self):
        return [self.c1, self.c2, self.c3, self.c4, self.fc5, self.fc6]
