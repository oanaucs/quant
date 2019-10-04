import tensorflow as tf

import numpy as np

from layers.conv_layer import conv2d
from layers.dense_layer import dense


class alexnet():
    def __init__(self, batch_size, num_classes, num_img_channels, var_dict=None):
        self.batch_size = batch_size
        self.num_classes = num_classes
        if var_dict is not None:
            self.create_pretrained_network_model(num_img_channels, var_dict)
        else:
            self.create_network_model(num_img_channels)

    def create_network_model(self, num_img_channels):
        self.conv1 = conv2d(in_depth=num_img_channels, out_depth=96, kernel_size=[11, 11], strides=[1, 4, 4, 1], padding='VALID',
            name='conv1')

        self.conv2_1 = conv2d(in_depth=48, out_depth=128, kernel_size=[5,5], padding='SAME', name='conv2_1')
        self.conv2_2 = conv2d(in_depth=48, out_depth=128, kernel_size=[5,5], padding='SAME', name='conv2_2')

        self.conv3 = conv2d(in_depth=256, out_depth=384, kernel_size=[3,3], padding='SAME', name='conv3')

        self.conv4_1 = conv2d(in_depth=192, out_depth=192, kernel_size=[3,3], padding='SAME', name='conv4_1')
        self.conv4_2 = conv2d(in_depth=192, out_depth=192, kernel_size=[3,3], padding='SAME', name='conv4_2')

        self.conv5_1 = conv2d(in_depth=192, out_depth=128, kernel_size=[3,3], padding='SAME', name='conv5_1')
        self.conv5_2 = conv2d(in_depth=192, out_depth=128, kernel_size=[3,3], padding='SAME', name='conv5_2')

        self.fc6 = dense(input_shape=[self.batch_size, 6 * 6 * 256],
            out_depth=4096, name='fc6')

        self.fc7 = dense(input_shape=[self.batch_size, 4096],
            out_depth=4096, name='fc7')

        self.fc8 = dense(input_shape=[self.batch_size, 4096], 
            out_depth=self.num_classes, name='fc8')


    # def create_pretrained_network_model(self, num_img_channels, var_dict):
    #     self.conv1 = conv2d(in_depth=num_img_channels, out_depth=96, kernel_size=[11, 11], strides=[1, 4, 4, 1], init_weights=var_dict['conv1'][0], padding='VALID',
    #         name='conv1')
    #     self.conv1.assign_bias_pretrained_weights(var_dict['conv1'][1])
        
    #     split_weights_c2 = np.split(var_dict['conv2'][0], 2, axis=3)

    #     self.conv2_1 = conv2d(in_depth=48, out_depth=128, kernel_size=[5,5], init_weights=split_weights_c2[0], padding='SAME', name='conv2_1')
    #     self.conv2_2 = conv2d(in_depth=48, out_depth=128, kernel_size=[5,5], init_weights=split_weights_c2[1], padding='SAME', name='conv2_2')

    #     split_bias_c2 = np.split(var_dict['conv2'][1], 2,)
    #     self.conv2_1.assign_bias_pretrained_weights(split_bias_c2[0])
    #     self.conv2_2.assign_bias_pretrained_weights(split_bias_c2[1])

    #     self.conv3 = conv2d(in_depth=256, out_depth=384, kernel_size=[3,3], init_weights=var_dict['conv3'][0], padding='SAME', name='conv3')
    #     self.conv3.assign_bias_pretrained_weights(var_dict['conv3'][1])

    #     split_weights_c4 = np.split(var_dict['conv4'][0], 2, axis=3)

    #     self.conv4_1 = conv2d(in_depth=192, out_depth=192, kernel_size=[3,3], init_weights=split_weights_c4[0], padding='SAME', name='conv4_1')
    #     self.conv4_2 = conv2d(in_depth=192, out_depth=192, kernel_size=[3,3], init_weights=split_weights_c4[0], padding='SAME', name='conv4_2')

    #     split_bias_c4 = np.split(var_dict['conv4'][1], 2,)
    #     self.conv4_1.assign_bias_pretrained_weights(split_bias_c4[0])
    #     self.conv4_2.assign_bias_pretrained_weights(split_bias_c4[1])

    #     split_weights_c5 = np.split(var_dict['conv5'][0], 2, axis=3)

    #     self.conv5_1 = conv2d(in_depth=192, out_depth=128, kernel_size=[3,3], init_weights=split_weights_c5[0], padding='SAME', name='conv5_1')
    #     self.conv5_2 = conv2d(in_depth=192, out_depth=128, kernel_size=[3,3], init_weights=split_weights_c5[0], padding='SAME', name='conv5_2')

    #     split_bias_c5 = np.split(var_dict['conv5'][1], 2,)
    #     self.conv5_1.assign_bias_pretrained_weights(split_bias_c5[0])
    #     self.conv5_2.assign_bias_pretrained_weights(split_bias_c5[1])


    #     self.fc6 = dense(input_shape=[self.batch_size, 6 * 6 * 256], init_weights=var_dict['fc6'][0],
    #         out_depth=4096, name='fc6')
    #     self.fc6.assign_bias_pretrained_weights(var_dict['fc6'][1])

    #     self.fc7 = dense(input_shape=[self.batch_size, 4096], init_weights=var_dict['fc7'][0],
    #         out_depth=4096, name='fc7')
    #     self.fc7.assign_bias_pretrained_weights(var_dict['fc7'][1])


    #     self.fc8 = dense(input_shape=[self.batch_size, 4096], 
    #         out_depth=self.num_classes, name='fc8')

    def forward_pass(self, images):
        conv1 = self.conv1.forward(images)

        maxpool1 = tf.contrib.layers.max_pool2d(
            conv1,
            [3, 3],
            2,
            # padding='VALID',
            scope='pool1')

        # print('m1', maxpool1.shape)

        split_inputs = tf.split(maxpool1, 2, axis=3)

        conv2_1 = self.conv2_1.forward(split_inputs[0])
        conv2_2 = self.conv2_2.forward(split_inputs[1])

        conv2 = tf.concat([conv2_1, conv2_2], axis=3)

        # print('conv2',conv2.shape)

        maxpool2 = tf.contrib.layers.max_pool2d(
            conv2,
            [3, 3],
            2,
            padding='VALID',
            scope='pool2')

        # print('m2', maxpool2.shape)

        conv3 = self.conv3.forward(maxpool2)
        print('conv3',conv3.shape)

        split_inputs = tf.split(conv3, 2, axis=3)

        print('split inputs', split_inputs[0].shape, split_inputs[1].shape)

        conv4_1 = self.conv4_1.forward(split_inputs[0])
        conv4_2 = self.conv4_2.forward(split_inputs[1])

        # conv4 = tf.concat([conv4_1, conv4_2], axis=3)

        # split_inputs = tf.split(conv4, 2, axis=3)

        conv5_1 = self.conv5_1.forward(conv4_1)
        print('conv5_1', conv5_1.shape)

        conv5_2 = self.conv5_2.forward(conv4_2)
        print('conv5_2', conv5_2.shape)

        conv5 = tf.concat([conv5_1, conv5_2], axis=3)

        print('conv5', conv5.shape)

        maxpool5 = tf.contrib.layers.max_pool2d(
            conv5,
            [3, 3],
            2,
            padding='VALID',
            scope='pool5')

        # print('m5', maxpool5.shape)

        flat_maxpool5 = tf.reshape(maxpool5, [self.batch_size, 6 * 6* 256]) 

        fc6 = self.fc6.forward(flat_maxpool5)
        # fc6_d = tf.contrib.layers.dropout(fc6, keep_prob=0.5, scope='dropout6')

        # print('fc6', fc6.shape)

        fc7 = self.fc7.forward(fc6)
        # fc7_d = tf.contrib.layers.dropout(fc7, keep_prob=0.5, scope='dropout7')

        # print('fc7', fc7.shape)

        logits = self.fc8.forward(fc7)

        # print('fc8', logits.shape)

        # logits = tf.squeeze(fc8, [1, 2], name='fc8/squeezed')

        predictions = tf.argmax(logits, axis=1)

        return logits, predictions
        return logits_values, predictions

    def backward_pass(self):
        pass

    def train(self, images):
        return self.forward_pass(images)

    def layers_to_compress(self):
        return [self.conv1, self.conv2_1, self.conv2_2,
                self.conv3, self.conv4_1, self.conv4_2,
                self.conv5_1, self.conv5_2, 
                self.fc6, self.fc7]