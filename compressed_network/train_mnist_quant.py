import numpy as np
import tensorflow as tf
import sys
sys.path.append('./../')

from layers.conv_layer import conv2d
from layers import dense_layer


from models.mnist_quant import CNN
from models.mnist_plain import mnist_plain

from datasets.dataset_factory import get_dataset\

import os

from tensorflow.python.ops import variable_scope as vs


#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'flowers', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './../tmp/flowers', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 28, 'Train image size')

tf.app.flags.DEFINE_string('checkpoint_dir', '../mnist_q_4e_0.001_pruned',
                           'Directory for saving and restoring checkpoints.')

#######################
# Training Flags #
#######################
tf.app.flags.DEFINE_integer(
    'num_epochs', 1,
    'Maximum number of epochs.')

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')


#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

FLAGS = tf.app.flags.FLAGS


def configure_learning_rate(num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_lr')


def configure_optimizer(learning_rate):
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    return optimizer

def gradients_cal(var_list, loss, lr, net):
    vs.get_variable_scope().reuse_variables()
    gradients_list = tf.gradients(xs=var_list, ys=loss)

    masks = [net.c1.prune_mask, net.c2.prune_mask, net.fc3.prune_mask, None, None, None, None, None]
    
    gradients_op_list = []
    gradients_mask_list = []
    # lr * gradients
    # gradients = [g for g in gradients_list]

    for i in range(len(var_list)):
        # print('gradient', gradients_list[i])
        # print('mask', masks[i])
        # collect the gradients op
        # apply mask
        if masks[i] is not None:
            gradients_mask = gradients_list[i] * lr * masks[i]
            gradients_mask_list.append(gradients_mask)
            new_var = tf.subtract(var_list[i], gradients_mask)
        else:
            new_var = tf.subtract(var_list[i], gradients_list[i] * lr)
        gradients_op = var_list[i].assign(new_var)
        gradients_op_list.append(gradients_op)
    
    return gradients_op_list, gradients_mask_list


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    g = tf.Graph()
    profiler = tf.profiler.Profiler(g)
    with g.as_default():
        run_meta = tf.RunMetadata()
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        # Create global_step
        global_step = tf.train.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset, num_classes, num_samples = get_dataset(
            'mnist',
            'train',
            './../tmp/mnist')

        print('dataset num classes', num_classes, num_samples)

        num_steps = int(num_samples / FLAGS.batch_size)

        dataset = dataset.repeat(FLAGS.num_epochs).shuffle(True).batch(FLAGS.batch_size)

        #########################
        # Load from the dataset #
        #########################
        # make iterator
        # TODO change iterator type
        iterator = dataset.make_one_shot_iterator()

        [images, labels] = iterator.get_next()

        images = tf.div(images, np.float32(255))

        images.set_shape([FLAGS.batch_size, 28, 28, 1])

        labels -= FLAGS.labels_offset

        onehot_labels = tf.one_hot(
            labels, num_classes - FLAGS.labels_offset)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        summaries.add(tf.summary.image('image', images))

        quant_net = CNN(FLAGS.batch_size, num_classes)
        logits, predictions = quant_net.train(images)

        loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits, label_smoothing=FLAGS.label_smoothing, weights=1.0))

        # Gather initial summaries.

        total_loss = tf.losses.get_total_loss()

        summaries.add(tf.summary.scalar('loss/%s' %
                                        total_loss.op.name, total_loss))

        streaming_accuracy = tf.contrib.metrics.accuracy(labels, predictions)
        summaries.add(tf.summary.scalar('accuracy', streaming_accuracy))

        # for end_point in end_points:
        #     x = end_points[end_point]
        # summaries.add(tf.summary.histogram('activations/' + end_point, x))
        #     summaries.add(tf.summary.scalar('sparsity/' + end_point,
        #                                   tf.nn.zero_fraction(x)))

        # for variable in tf.contrib.framework.get_model_variables():
        #     summaries.add(tf.summary.histogram(variable.op.name, variable))

        model_variables = tf.contrib.framework.get_model_variables()

        variables_to_train = tf.trainable_variables()
        print(variables_to_train)

        learning_rate = configure_learning_rate(num_samples, global_step)s
        optimizer = configure_optimizer(learning_rate)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        # gradient_vars = optimizer.compute_gradients(
        #     loss_op, variables_to_train)

        # gradients = [grad for grad, var in gradient_vars]

        # train_step = optimizer.apply_gradients(
        #     gradient_vars, global_step=global_step)

        train_step, _ = gradients_cal(variables_to_train, total_loss, 0.03, quant_net)

        layers_to_compress = quant_net.layers_as_list()

        layers_to_compress_names = []
        for layer in layers_to_compress:
            layers_to_compress_names.append(layer.name)

        saver = tf.train.Saver()

        # with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.checkpoint_dir,
        #                                        config=config,
        #                                        save_summaries_steps=10) as mon_sess:
        with tf.Session() as mon_sess:
            mon_sess.run(tf.global_variables_initializer())

            for i in range(0, 100):
                for j in range(0, 100):

                    _, loss = mon_sess.run([train_step, total_loss])

                    print('loss', loss)
                    print(' ')
                    print('accuracy', mon_sess.run(streaming_accuracy))
                    print(' ')

            saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'init_model.ckpt'))

        #     saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        #     # # # ###########
        #     # # # PRUNE #
        #     for i in range(0, 2):
        #         for layer in layers_to_compress:
        #             layer.prune_weights(mon_sess)

        #         last_c1_values = mon_sess.run(layers_to_compress[0].weights)
        #         print('last c1 values', last_c1_values)


        #         saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'pruned_model.ckpt'))

        #         saver.restore(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'pruned_model.ckpt'))

        #         new_values = mon_sess.run(layers_to_compress[0].weights)
        #         # print('new_values', new_values)

        #         gradients_list_op, gradients_masks_list = gradients_cal(var_list=variables_to_train, loss=total_loss, lr=0.03, net= quant_net)
        #         print(gradients_list_op)

        #         _, prune_loss = mon_sess.run([train_step, total_loss])
        #         print('last prune loss', prune_loss)
        #         # print('accuracy', mon_sess.run(streaming_accuracy))


        #         last_c1_values = mon_sess.run(layers_to_compress[0].weights)
        #         print('last c1 values', last_c1_values)
            

                
            # # ############
            # # # QUANTIZE #
            # for i in range(0, 1000):
            #     for layer in layers:
            #         layer.quantize_weights(mon_sess)

            #     # retrain
            #     eval_images = mon_sess.run(images)
            #     eval_labels = mon_sess.run(labels)

            #     gradient_values, quantization_loss = mon_sess.run(
            #         [gradients, total_loss], feed_dict={images_var: eval_images, labels: eval_labels})

            #     # print('quantization loss', quantization_loss)
            #     # print('c1 values')
            #     # c1_values = mon_sess.run(layers[0].weights)
            #     # print(c1_values)

            #     # print('difference')
            #     # print(last_c1_values - c1_values)

            #     quantized_gradients = {}
            #     for i in range(len(gradients)):
            #         quantized_gradients[gradients[i]] = layers[i].quantize_gradients(
            #             gradient_values[i])

            #     mon_sess.run(train_step, feed_dict=quantized_gradients)


if __name__ == '__main__':
    main()
