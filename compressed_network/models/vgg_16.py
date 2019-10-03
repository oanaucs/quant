import tensorflow as tf

import numpy as np

from layers.conv_layer import conv2d
from layers.dense_layer import dense


class vgg_16():
    def __init__(self, images_placeholder, num_classes, ckpt_var_dict=None, pruning_threshold=0.05):
        self.input_shape = images_placeholder.get_shape().as_list()
        self.pruning_threshold = pruning_threshold
        # print('input shape', images_placeholder.get_shape())
        # self.iterator =

        self.create_network_layers(num_classes)
        if ckpt_var_dict is not None:
            self.init_layers_from_checkpoint(ckpt_var_dict)
        # self.setup_train()

    def init_layers_from_checkpoint(self, var_dict):
        layers_as_list = self.layers_as_list()

        for var_key, var_value in var_dict.items():
            var_name = var_key.split('/')
            if ('conv' in var_key):
                layer_name = var_name[0] + '/' + var_name[1] + '/' + var_name[2]
            elif ('fc' in var_key):
                if ('fc8' not in var_key):
                    layer_name = var_name[0] + '/' + var_name[1]
                else:
                    layer_name = None
            else:
                layer_name = None

            if layer_name:
                for layer in layers_as_list:
                    if (layer.name.find(layer_name) > -1):
                        if ('biases' in var_name[-1]):
                            layer.assign_bias_weights(var_value)
                        else:
                            layer.assign_weights(var_value)
        


    def create_network_layers(self, num_classes):
        self.conv1_1 = conv2d(in_depth=1, out_depth=64, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv1/conv1_1')
        self.conv1_2 = conv2d(in_depth=64, out_depth=64, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv1/conv1_2')

        self.conv2_1 = conv2d(in_depth=64, out_depth=128, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv2/conv2_1')
        self.conv2_2 = conv2d(in_depth=128, out_depth=128, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv2/conv2_2')

        self.conv3_1 = conv2d(in_depth=128, out_depth=256, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv3/conv3_1')
        self.conv3_2 = conv2d(in_depth=256, out_depth=256, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv3/conv3_2')
        self.conv3_3 = conv2d(in_depth=256, out_depth=256, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv3/conv3_3')

        self.conv4_1 = conv2d(in_depth=256, out_depth=512, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv4/conv4_1')
        self.conv4_2 = conv2d(in_depth=512, out_depth=512, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv4/conv4_2')
        self.conv4_3 = conv2d(in_depth=512, out_depth=512, 
            pruning_threshold=self.pruning_threshold, name='vgg16/conv4/conv4_3')

        self.conv5_1 = conv2d(in_depth=512, out_depth=512, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv5/conv5_1')
        self.conv5_2 = conv2d(in_depth=512, out_depth=512, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv5/conv5_2')
        self.conv5_3 = conv2d(in_depth=512, out_depth=512, 
            pruning_threshold=self.pruning_threshold, name='vgg_16/conv5/conv5_3')

        self.fc6 = conv2d(in_depth=512, out_depth=4096, 
            pruning_threshold=self.pruning_threshold, kernel_size=[
            7, 7], padding='VALID', name='vgg_16/fc6')

        self.fc7 = conv2d(in_depth=4096, out_depth=4096, 
            pruning_threshold=self.pruning_threshold, kernel_size=[
            1, 1], name='vgg_16/fc7')

        self.fc8 = conv2d(in_depth=4096, out_depth=num_classes,
            pruning_threshold=self.pruning_threshold, kernel_size=[
            1, 1], name='vgg_16/fc8')

    def forward_pass(self, images):
        conv1_1 = self.conv1_1.forward(images)
        # print('conv11', conv1_1.shape)
        
        conv1_2 = self.conv1_2.forward(conv1_1)
        # print('conv12', conv1_2.shape)

        maxpool1 = tf.contrib.layers.max_pool2d(conv1_2, [2, 2], scope='pool1')
        # print('maxpool1', maxpool1.shape)

        conv2_1 = self.conv2_1.forward(maxpool1)

        # print('conv21', conv2_1.shape)
        conv2_2 = self.conv2_2.forward(conv2_1)

        # print('conv22', conv2_2.shape)

        maxpool2 = tf.contrib.layers.max_pool2d(conv2_2, [2, 2], scope='pool2')
        # print('maxpool2', maxpool2.shape)

        conv3_1 = self.conv3_1.forward(maxpool2)

        # print('conv31', conv3_1.shape)

        conv3_2 = self.conv3_2.forward(conv3_1)
        # print('conv32', conv3_2.shape)

        conv3_3 = self.conv3_3.forward(conv3_2)
        # print('conv32', conv3_2.shape)

        maxpool3 = tf.contrib.layers.max_pool2d(conv3_3, [2, 2], scope='pool3')
        # print('maxpool3', maxpool3.shape)

        conv4_1 = self.conv4_1.forward(maxpool3)

        # print('conv41', conv4_1.shape)
        conv4_2 = self.conv4_2.forward(conv4_1)
        # print('conv42', conv4_2.shape)

        conv4_3 = self.conv4_3.forward(conv4_2)
        # print('conv32', conv3_2.shape)

        maxpool4 = tf.contrib.layers.max_pool2d(conv4_3, [2, 2], scope='pool4')
        # print('maxpool4', maxpool4.shape)

        conv5_1 = self.conv5_1.forward(maxpool4)

        # print('conv51', conv5_1.shape)
        conv5_2 = self.conv5_2.forward(conv5_1)

        # print('conv52', conv5_2.shape)

        conv5_3 = self.conv5_3.forward(conv5_2)

        maxpool5 = tf.contrib.layers.max_pool2d(conv5_3, [2, 2], scope='pool5')
        # print('maxpool5', maxpool5.shape)

        # fc6 = self.fc6.forward(maxpool3)
        fc6 = self.fc6.forward(maxpool5)

        fc6_d = tf.contrib.layers.dropout(fc6, keep_prob=0.5, scope='dropout6')

        fc7 = self.fc7.forward(fc6_d)
        
        fc7_d = tf.contrib.layers.dropout(fc7, keep_prob=0.5, scope='dropout7')

        fc8 = self.fc8.forward(fc7_d)

        logits = tf.squeeze(fc8, [1, 2], name='fc8/squeezed')

        predictions = tf.argmax(logits, axis=1)

        return logits, predictions

    def backward_pass(self):
        pass

    def train(self, images):
        return self.forward_pass(images)

    def layers_as_list(self):
        return [self.conv1_1, self.conv1_2,
                self.conv2_1, self.conv2_2,
                self.conv3_1, self.conv3_2, self.conv3_3,
                self.conv4_1, self.conv4_2, self.conv4_3,
                self.conv5_1, self.conv5_2, self.conv5_3,
                self.fc6, self.fc7, self.fc8]

        # return [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2,
        # self.conv3_1, self.conv3_2,
        # self.fc6, self.fc7, self.fc8]

    def weights_as_list(self):
        return [self.conv1_1.weights, self.conv1_2.weights,
                self.conv2_1.weights, self.conv2_2.weights,
                self.conv3_1.weights, self.conv3_2.weights,
                self.conv4_1.weights, self.conv4_2.weights,
                self.conv4_1.weights, self.conv4_2.weights,
                self.conv5_1.weights, self.conv5_2.weights,
                self.fc6.weights, self.fc7.weights,
                self.fc8.weights]

        # return [self.conv1_1.weights, self.conv1_2.weights,
        # self.conv2_1.weights, self.conv2_2.weights,
        # self.conv3_1.weights, self.conv3_2.weights,
        # self.fc6.weights, self.fc7.weights,
        # self.fc8.weights]
