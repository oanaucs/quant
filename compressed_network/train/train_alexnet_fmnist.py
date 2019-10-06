import sys
sys.path.append('./../')
sys.path.append('./../../')

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from layers.conv_layer import conv2d
from layers import dense_layer

from models.alexnet import alexnet

from utils.prune_utils import increase_sparsity_level

from datasets.dataset_factory import get_dataset
from preprocessing import preprocessing_factory

#################
# Dataset Flags #
#################

tf.app.flags.DEFINE_string(
    'dataset_name', 'fashion_mnist', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '.\\..\\..\\tmp\\fashion_mnist', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 227, 'Train image size')

##################
# Training Flags #
##################

tf.app.flags.DEFINE_string('checkpoint_dir', '.\\..\\..\\trained_models\\alexnet_quant',
                           'Directory for saving and restoring checkpoints.')

tf.app.flags.DEFINE_integer(
    'batch_size', 128, 'The number of samples in each batch.')
    
tf.app.flags.DEFINE_integer(
    'num_epochs', 1500,
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
    'linear',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.03, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0003,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 100,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#################
# Pruning Flags #
#################

tf.app.flags.DEFINE_integer(
    'num_pruning_steps', 61,
    'Number of pruning steps.')

tf.app.flags.DEFINE_integer(
    'num_pruning_retrain_steps', 250,
    'Number of retrain steps after_pruning.')

tf.app.flags.DEFINE_float(
    'pruning_threshold', None,
    'Pruning threshold. If set to None, adaptive sparsity level will be used.')

tf.app.flags.DEFINE_float(
    'init_sparsity_level', 0.8,
    'Initial sparsity level.')

tf.app.flags.DEFINE_float(
    'max_sparsity_level', 0.99,
    'Maximum sparsity level. Depending on the number of steps can be achieved or not')

tf.app.flags.DEFINE_float(
    'sparsity_increase_step', 0.002,
    'Step for increasing the sparsity level after the num_steps_sparsity_increase.')

######################
# Quantization Flags #
######################

tf.app.flags.DEFINE_integer(
    'num_quant_clusters', 16,
    'Number of quantization steps.')

tf.app.flags.DEFINE_integer(
    'num_quant_steps', 1,
    'Number of quantization steps.')

tf.app.flags.DEFINE_integer(
    'num_quant_retrain_steps', 10,
    'Number of steps for retraining after quantization.')

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
    elif FLAGS.learning_rate_decay_type == 'linear':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate=FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name=None)


def apply_gradients(eval_grad_list, var_list, lr): 
    updated_weights = []

    for i in range(len(var_list)):
        updated_weights.append(tf.subtract(var_list[i], eval_grad_list[i] * lr))

    return updated_weights


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        # Create global_step
        global_step = tf.train.create_global_step()
        global_step_count = 0
        global_step_ph = tf.placeholder(tf.int64)
        global_step_assign_op = global_step.assign(global_step_ph)
        # create summary writer
        summary_writer = tf.contrib.summary.create_file_writer(FLAGS.checkpoint_dir, flush_millis=10000)

        #####################################
        # Select the preprocessing function #
        #####################################
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            'lenet',
            is_training=True)

        ######################
        # Select the dataset #
        ######################
        dataset, num_classes, num_samples = get_dataset(
            FLAGS.dataset_name,
            FLAGS.dataset_split_name,
            FLAGS.dataset_dir)

        print('dataset num classes', num_classes, num_samples)

        num_steps = int(num_samples / FLAGS.batch_size)

        dataset = dataset.map(lambda image, label: (image_preprocessing_fn(
            image, FLAGS.train_image_size, FLAGS.train_image_size), label))

        dataset = dataset.repeat(FLAGS.num_epochs).shuffle(True).batch(FLAGS.batch_size)

        #########################
        # Load from the dataset #
        #########################
        # make iterator
        # TODO change iterator type
        iterator = dataset.make_one_shot_iterator()

        [images, labels] = iterator.get_next()

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.image('image', images))

        onehot_labels = tf.one_hot(
            labels, num_classes)

        ######################
        # Create the network #
        ######################
        net = alexnet(FLAGS.batch_size, num_classes, 1)
        logits, predictions = net.train(images)

        # list of layers to compress
        layers_to_compress = net.layers_to_compress()

        # create loss
        loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits, label_smoothing=FLAGS.label_smoothing, weights=1.0))

        total_loss = tf.losses.get_total_loss()

        summaries.add(tf.summary.scalar('loss/%s' %
                                        total_loss.op.name, total_loss))

        # define learning rate
        learning_rate = configure_learning_rate(num_samples, tf.train.get_global_step())
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        # define accuracy 
        acc_op, acc_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

        # gather summaries
        summaries.add(tf.summary.scalar('accuracy', acc_op))

        for layer in layers_to_compress:
            summaries.add(tf.summary.histogram('activations/' + layer.name, layer.weights))
            summaries.add(tf.summary.scalar('sparsity/' + layer.name,
                tf.nn.zero_fraction(layer.weights)))

        # merge summaries
        merged_summary_op = tf.summary.merge_all()

        #######################
        # Define backprop ops #
        #######################
        # allow variable reuse
        variable_scope.get_variable_scope().reuse_variables()
        variables_to_train = tf.trainable_variables()
        print('trainable variables', variables_to_train)

        grads_placeholders = [tf.placeholder(tf.float32, shape=v.shape) for v in variables_to_train]
        compute_weight_updates_op = [variables_to_train[i].assign(tf.subtract(variables_to_train[i], grads_placeholders[i])) for i in range(0, len(variables_to_train))]

        sparse_vars = []
        for var in variables_to_train:
            sparse_vars.append(tf.contrib.layers.dense_to_sparse(var, scope=var.name.split(':')[0]))

        # gradient computation op
        gradients_list = tf.gradients(xs=variables_to_train, ys=total_loss)

        ################
        # Create saver #
        ################
        saver = tf.train.Saver(variables_to_train, max_to_keep=200)
        writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, tf.get_default_graph())
        last_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        with tf.Session() as mon_sess:
            mon_sess.run(tf.global_variables_initializer())
            mon_sess.run(tf.local_variables_initializer())

            mon_sess.graph.finalize()

            if last_ckpt is not None:
                saver.restore(mon_sess, last_ckpt)
                # get global step
                global_step_count = int(last_ckpt.split('-')[-1])
                mon_sess.run(global_step_assign_op, feed_dict={global_step_ph: global_step_count})

            vars_values = []
            for op in variables_to_train:
                vars_values.append(mon_sess.run(op))

            np.save(os.path.join(FLAGS.checkpoint_dir, 'entire'), vars_values)


            ####################
            # INITIAL TRAINING #
            # ####################

            # for epoch in range(0, FLAGS.num_epochs):
            #     for step in range(0, num_steps):
            #         # run gradients
            #         eval_grad = mon_sess.run(gradients_list)

            #         # assign weights
            #         for i in range(0, len(variables_to_train)):
            #             mon_sess.run(compute_weight_updates_op[i], feed_dict={grads_placeholders[i]: eval_grad[i] * mon_sess.run(learning_rate)})
            #         # compute loss and accuracy
            #         loss, summary = mon_sess.run([total_loss, merged_summary_op])
            #         accuracy, _ = mon_sess.run([acc_op, acc_update_op])

            #         print('training loss', loss, 'accuracy', accuracy)

            #         # update global_step
            #         global_step_count += 1
            #         mon_sess.run(global_step_assign_op, feed_dict={global_step_ph: global_step_count})

            #         # save intermediate model
            #         if (step % 20 == 0):
            #             writer.add_summary(summary, mon_sess.run(global_step))
            #             saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-' + str(mon_sess.run(global_step))))
                        
            # # save final model
            # saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'init_model.ckpt'))

            #########
            # PRUNE #
            #########
            # current_sparsity_level = FLAGS.init_sparsity_level
            # for prune_step in range(0, FLAGS.num_pruning_steps):
            #     # restore saved model
            #     saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            #     # compute current threshold or sparsity level
            #     if FLAGS.pruning_threshold is None:
            #         if (prune_step > 0):
            #             current_sparsity_level = increase_sparsity_level(current_sparsity_level, FLAGS.max_sparsity_level, FLAGS.sparsity_increase_step)
                
            #     # quantize layers
            #     for layer in layers_to_compress:
            #         layer.prune_weights(mon_sess, FLAGS.pruning_threshold, current_sparsity_level)

            #     # last_c1_values = mon_sess.run(layers_to_compress[0].weights)
            #     # print('last c1 values', last_c1_values)

            #     # save current weights
            #     saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'pruned_model.ckpt'))
            #     # restore for retraining
            #     saver.restore(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'pruned_model.ckpt'))

            #     # new_values = mon_sess.run(layers_to_compress[0].weights))

            #     # retrain
            #     for step in range(0, FLAGS.num_pruning_retrain_steps):
            #         # run gradients
            #         eval_grad = mon_sess.run(gradients_list)
            #         # prune gradients
                    # # eval_grad[0] = net.conv1.prune_gradients(eval_grad[0])
                    # # eval_grad[1] = net.conv2_1.prune_gradients(eval_grad[1])
                    # # eval_grad[2] = net.conv2_2.prune_gradients(eval_grad[2])
                    # # eval_grad[3] = net.conv3.prune_gradients(eval_grad[3])
                    # eval_grad[4] = net.conv4_1.prune_gradients(eval_grad[4])
                    # eval_grad[5] = net.conv4_2.prune_gradients(eval_grad[5])
                    # eval_grad[6] = net.conv5_1.prune_gradients(eval_grad[6])
                    # eval_grad[7] = net.conv5_2.prune_gradients(eval_grad[7])
                    # eval_grad[8] = net.fc6.prune_gradients(eval_grad[8])
                    # eval_grad[9] = net.fc7.prune_gradients(eval_grad[9])

            #         lr = FLAGS.learning_rate * 0.95 ** (step / 5)

            #         # assign weights
            #         for i in range(0, len(variables_to_train)):
            #             mon_sess.run(compute_weight_updates_op[i], feed_dict={grads_placeholders[i]: eval_grad[i] * lr})
                    
            #         # compute loss
            #         prune_loss, summary = mon_sess.run([total_loss, merged_summary_op])
            #         accuracy, _ = mon_sess.run([acc_op, acc_update_op])

            #         print('prune loss', prune_loss, 'accuracy', accuracy)

            #         # update global_step
            #         global_step_count += 1
            #         mon_sess.run(global_step_assign_op, feed_dict={global_step_ph: global_step_count})

            #         # last_c1_values = mon_sess.run(layers_to_compress[0].weights)
            #         # print('last c1 values', last_c1_values)

            #         # save pruned model
            #     if (prune_step % 10 == 0):
            #         # if (step % 20 == 0):
            #         writer.add_summary(summary, mon_sess.run(global_step))
            #         saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, str(current_sparsity_level) + '_pruned_model.ckpt-' + str(mon_sess.run(global_step))))
                    
            ############
            # QUANTIZE #
            ############
            # layers_to_compress.append(net.fc8)
            # for quant_step in range(0, 1):
            #     # restore model model
            #     saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

            #     start_quant_time = time.time()
                
            #     for layer in layers_to_compress:
            #         layer.quantize_weights(mon_sess, FLAGS.num_quant_clusters)

            #     end_quant_time = time.time() - start_quant_time

            #     # last_c1_values = mon_sess.run(layers_to_compress[0].weights)
            #     # print('last c1 values', last_c1_values)

            #     # save weights
            #     saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'quant_model.ckpt'))
            #     # restore for retraining
            #     saver.restore(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'quant_model.ckpt'))

            #     # last_c1_values = mon_sess.run(layers_to_compress[0].weights)
            #     # print('last c1 values', last_c1_values)

            #     quant_retrain_step_times = []

            #     for step in range(0, FLAGS.num_quant_retrain_steps):
            #         start_time = time.time()
            #         # get gradients
            #         eval_grad = mon_sess.run(gradients_list)
            #         # quantize gradients
            #         eval_grad[4] = net.conv4_1.quantize_gradients(eval_grad[4])
            #         eval_grad[5] = net.conv4_2.quantize_gradients(eval_grad[5])
            #         eval_grad[6] = net.conv5_1.quantize_gradients(eval_grad[6])
            #         eval_grad[7] = net.conv5_2.quantize_gradients(eval_grad[7])
            #         eval_grad[8] = net.fc6.quantize_gradients(eval_grad[8])
            #         eval_grad[9] = net.fc7.quantize_gradients(eval_grad[9])
            #         eval_grad[10] = net.fc8.quantize_gradients(eval_grad[10])

            #         lr = FLAGS.learning_rate * 0.95 ** 1

            #         # update weights
            #         for i in range(0, len(variables_to_train)):
            #             mon_sess.run(compute_weight_updates_op[i], feed_dict={grads_placeholders[i]: eval_grad[i] * 0.00001})

            #         # compute loss and accuracy
            #         quant_loss, summary = mon_sess.run([total_loss, merged_summary_op])
            #         quant_retrain_step_times.append(time.time() - start_time)
            #         accuracy, _ = mon_sess.run([acc_op, acc_update_op])

            #         global_step_count += 1
            #         mon_sess.run(global_step_assign_op, feed_dict={global_step_ph: global_step_count})

            #         print('quant loss', quant_loss, 'accuracy', accuracy)
                
            #     writer.add_summary(summary, mon_sess.run(global_step))
            #     saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + '_quant_model.ckpt-' + str(mon_sess.run(global_step))))

            # for layer in layers_to_compress:
            #     np.save(os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + '_codebook_' + layer.name), layer.centroids)
            # np.save(os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + 'quant_time'), end_quant_time)
            # np.save(os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + 'avg_quant_step'), np.sum(quant_retrain_step_times) / FLAGS.num_quant_retrain_steps)



if __name__ == '__main__':
    main()
