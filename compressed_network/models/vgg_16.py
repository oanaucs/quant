import tensorflow as tf

import numpy as np

from layers.conv_layer import conv2d
from layers.dense_layer import dense


class vgg_16():
    def __init__(self, num_classes, num_channels):
        self.create_network_layers(num_classes)
       

    def init_layers_from_checkpoint(self, session, var_dict):
        layers_as_list = self.layers_as_list()

        for var_key, var_value in var_dict.items():
            var_name = var_key.split('/')
            if ('conv' in var_key):
                layer_name = var_name[0] + '/' + var_name[1] + '/' + var_name[2]
            # elif ('fc' in var_key):
            #     if ('fc8' not in var_key):
            #         layer_name = var_name[0] + '/' + var_name[1]
            #     else:
            #         layer_name = None
            else:
                layer_name = None

            if layer_name:
                for layer in layers_as_list:
                    if (layer.name.find(layer_name) > -1):
                        if ('biases' in var_name[-1]):
                            layer.assign_bias_weights(var_value)
                            print('assigning bias', layer.name)
                        else:
                            layer.assign_weights(session, var_value)
                            print('assigning weight', layer.name)
        


    def create_network_layers(self, num_classes):
        self.conv1_1 = conv2d(in_depth=3, out_depth=64, 
            name='vgg_16/conv1/conv1_1', trainable=False)
        self.conv1_2 = conv2d(in_depth=64, out_depth=64, 
            name='vgg_16/conv1/conv1_2', trainable=False)

        self.conv2_1 = conv2d(in_depth=64, out_depth=128, 
            name='vgg_16/conv2/conv2_1', trainable=False)
        self.conv2_2 = conv2d(in_depth=128, out_depth=128, 
            name='vgg_16/conv2/conv2_2', trainable=False)

        self.conv3_1 = conv2d(in_depth=128, out_depth=256, 
            name='vgg_16/conv3/conv3_1', trainable=False)
        self.conv3_2 = conv2d(in_depth=256, out_depth=256, 
            name='vgg_16/conv3/conv3_2', trainable=False)
        self.conv3_3 = conv2d(in_depth=256, out_depth=256, 
            name='vgg_16/conv3/conv3_3', trainable=False)

        self.conv4_1 = conv2d(in_depth=256, out_depth=512, 
            name='vgg_16/conv4/conv4_1', trainable=False)
        self.conv4_2 = conv2d(in_depth=512, out_depth=512, 
            name='vgg_16/conv4/conv4_2', trainable=False)
        self.conv4_3 = conv2d(in_depth=512, out_depth=512, 
            name='vgg16/conv4/conv4_3', trainable=False)

        self.conv5_1 = conv2d(in_depth=512, out_depth=512, 
            name='vgg_16/conv5/conv5_1', trainable=False)
        self.conv5_2 = conv2d(in_depth=512, out_depth=512, 
            name='vgg_16/conv5/conv5_2', trainable=False)
        self.conv5_3 = conv2d(in_depth=512, out_depth=512, 
            name='vgg_16/conv5/conv5_3', trainable=False)

        self.fc6 = conv2d(in_depth=512, out_depth=4096, 
            kernel_size=[7, 7], padding='VALID', name='vgg_16/fc6', trainable=True)

        self.fc7 = conv2d(in_depth=4096, out_depth=4096, 
            kernel_size=[1, 1], name='vgg_16/fc7', trainable=True)

        self.fc8 = conv2d(in_depth=4096, out_depth=10,
            kernel_size=[1, 1], name='vgg_16/fc8', trainable=True)

    def forward_pass(self, images):
        conv1_1 = self.conv1_1.forward(images)
        
        conv1_2 = self.conv1_2.forward(conv1_1)

        maxpool1 = tf.contrib.layers.max_pool2d(
            conv1_2,
            [2, 2], 
            padding='SAME',
            scope='pool1')

        conv2_1 = self.conv2_1.forward(maxpool1)

        conv2_2 = self.conv2_2.forward(conv2_1)

        maxpool2 = tf.contrib.layers.max_pool2d(
            conv2_2,
            [2, 2],
            padding='SAME',
            scope='pool2')

        conv3_1 = self.conv3_1.forward(maxpool2)

        conv3_2 = self.conv3_2.forward(conv3_1)

        conv3_3 = self.conv3_3.forward(conv3_2)

        maxpool3 = tf.contrib.layers.max_pool2d(
            conv3_3, 
            [2, 2],
            padding='SAME',
            scope='pool3')

        conv4_1 = self.conv4_1.forward(maxpool3)

        conv4_2 = self.conv4_2.forward(conv4_1)

        conv4_3 = self.conv4_3.forward(conv4_2)

        maxpool4 = tf.contrib.layers.max_pool2d(conv4_3, [2, 2], scope='pool4')

        conv5_1 = self.conv5_1.forward(maxpool4)

        conv5_2 = self.conv5_2.forward(conv5_1)

        conv5_3 = self.conv5_3.forward(conv5_2)

        maxpool5 = tf.contrib.layers.max_pool2d(
            conv5_3,
            [2, 2],
            padding='SAME',
            scope='pool5')


        fc6 = self.fc6.forward(maxpool5)
        print('fc6', fc6.shape)

        fc6_d = tf.contrib.layers.dropout(fc6, keep_prob=0.5, scope='dropout6')

        fc7 = self.fc7.forward(fc6_d)
        
        fc7_d = tf.contrib.layers.dropout(fc7, keep_prob=0.5, scope='dropout7')

        fc8 = self.fc8.forward(fc7_d)

        print('logits unsqueezed', fc8.shape)

        logits = tf.squeeze(fc8, [1, 2], name='fc8/squeezed')
        print('logits squeezed', logits.shape)

        predictions = tf.argmax(logits, axis=1)

        return logits, predictions

    def backward_pass(self):
        pass

    def train(self, images):
        return self.forward_pass(images)

    def layers_to_compress(self):
        return [self.fc6, self.fc7, self.fc8]

    def layers_as_list(self):
        return [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2,
                self.conv3_1, self.conv3_2, self.conv3_3, 
                self.conv4_1, self.conv4_2, self.conv4_3,
                self.conv5_1, self.conv5_2, self.conv5_3,
                self.fc6, self.fc7, self.fc8]