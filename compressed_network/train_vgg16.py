import numpy as np
import tensorflow as tf
import sys
sys.path.append('./../')

import os

from models.vgg_16 import vgg_16

from tensorflow.python.ops import variable_scope

from preprocessing import preprocessing_factory
from datasets.dataset_factory import get_dataset

from load_pretrained_ckpt.inspect_ckpt import load_variables

#################
# Dataset Flags #
#################

tf.app.flags.DEFINE_string(
    'dataset_name', 'mnist', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './../tmp/mnist', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

#######################
# Training Flags #
#######################

tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 
    # '../slim_pretrained_ckpt/vgg_16.ckpt',
    None,
    'Path to pretrained model for warm start training. If None, cold start.')

tf.app.flags.DEFINE_string('checkpoint_dir', '../trained_models/vgg_fmnist/',
                           'Directory for saving and restoring checkpoints.')

tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_epochs', 10,
    'Maximum number of epochs.')

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float(
    'opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

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
    'num_epochs_per_decay', 0.5,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#################
# Pruning Flags #
#################

tf.app.flags.DEFINE_integer(
    'num_pruning_steps', 101,
    'Number of pruning steps.')
    
tf.app.flags.DEFINE_integer(
    'num_pruning_retrain_steps', 100,
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
    'num_quant_clusters', 256,
    'Number of quantization steps.')

tf.app.flags.DEFINE_integer(
    'num_quant_steps', 10,
    'Number of quantization steps.')

tf.app.flags.DEFINE_integer(
    'num_quant_retrain_steps', 50,
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

def apply_gradients(eval_grad_list, var_list, lr): 
    updated_weights = []

    for i in range(len(var_list)):
        updated_weights.append(tf.subtract(var_list[i], eval_grad_list[i] * lr))

    return updated_weights


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    g = tf.Graph()
    profiler = tf.profiler.Profiler(g)
    with g.as_default():
        run_meta = tf.RunMetadata()
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        global_step = tf.train.create_global_step()
        global_step_count = 0
        global_step_ph = tf.placeholder(tf.int64)
        global_step_assign_op = global_step.assign(global_step_ph)
        # create summary writer

        #####################################
        # Select the preprocessing function #
        #####################################
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            'vgg_16',
            is_training=True)

        train_image_size = 224

        ######################
        # Select the dataset #
        ######################
        dataset, num_classes, num_samples = get_dataset(
            'fashion_mnist',
            'train',
            './../tmp/fashion_mnist')

        num_steps = int(num_samples / FLAGS.batch_size)
        print('num steps', num_steps) 

        # dataset = dataset.apply(tf.contrib.data.ignore_errors())

        dataset = dataset.map(lambda image, label: (image_preprocessing_fn(
            image, train_image_size, train_image_size), label))

        dataset = dataset.repeat(FLAGS.num_epochs).shuffle(
            True).batch(FLAGS.batch_size)

        #########################
        # Load from the dataset #
        #########################
        iterator = dataset.make_one_shot_iterator()

        [images, labels] = iterator.get_next()
        print('imags shape', images.shape)
        images.set_shape([FLAGS.batch_size, 224, 224, 1])

        images = tf.concat([images, images, images], axis=3)
        print('stacked imags shape', images.shape)



        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.image('image', images))

        onehot_labels = tf.one_hot(
            labels, num_classes)

        print('onehot labels', onehot_labels.shape)

        #################################
        # Configure pretrained network  #
        #################################

        restore_model_op = None
        # if pretrained model is specified, init from checkpoint
        if FLAGS.pretrained_model_ckpt_path:
            # load variables from ckpt 
            variables_to_restore = load_variables(FLAGS.pretrained_model_ckpt_path)
        else:
            variables_to_restore = None

        ######################
        # Select the network #
        ######################
        net = vgg_16(images, num_classes)

        logits, predictions = net.train(images)

        print('logits', logits.shape)

        tf.summary.tensor_summary('predictions', predictions)

        layers_to_compress = net.layers_to_compress()

        loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits, label_smoothing=FLAGS.label_smoothing, weights=1.0))

        total_loss = tf.losses.get_total_loss()

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

        print('trainable vars', variables_to_train)

        # updated_weights_placeholders = [tf.placeholder(tf.float32, shape=v.shape) for v in variables_to_train]
        # print('weights placeholders', updleaated_weights_placeholders)
        # assign_updated_weights_ops = [v.assign(p) for (v, p) in zip(variables_to_train, updated_weights_placeholders)]

        grads_placeholders = [tf.placeholder(tf.float32, shape=v.shape) for v in variables_to_train]
        compute_weight_updates_op = [variables_to_train[i].assign(tf.subtract(variables_to_train[i], grads_placeholders[i])) for i in range(0, len(variables_to_train))]
        
        # gradient computation op
        gradients_list = tf.gradients(xs=variables_to_train, ys=total_loss)

        ######################
        # Start the training #
        ######################
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

            if variables_to_restore is not None:
                net.init_layers_from_checkpoint(mon_sess, variables_to_restore)
                saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'pretrained_init_model.ckpt'))

                saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'pretrained_init_model.ckpt'))

                # assert np.array_equal(mon_sess.run(net.conv1_1.weights), variables_to_restore['vgg_16/conv1/conv1_1/weights'])


            elif last_ckpt is not None:
                saver.restore(mon_sess, last_ckpt)
                # get global step
                mon_sess.run(global_step_assign_op, feed_dict={global_step_ph: 0})


            ####################
            # INITIAL TRAINING #
            # ####################

            # for epoch in range(0, 500):
            #     print('epoch', epoch)
            #     for step in range(0, 100):
            #         # run gradients
            #         eval_grad = mon_sess.run(gradients_list)

            #         lr = FLAGS.learning_rate *0.9**(epoch/100)
                    
            #         # assign weights
            #         for i in range(0, len(variables_to_train)):
            #             mon_sess.run(compute_weight_updates_op[i], feed_dict={grads_placeholders[i]: eval_grad[i] * lr})
            #         # compute loss and accuracy
            #         loss, summary = mon_sess.run([total_loss, merged_summary_op])
            #         accuracy, _ = mon_sess.run([acc_op, acc_update_op])

            #         print('step', step, 'training loss', loss, 'accuracy', accuracy)

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

            print('vars to train', variables_to_train)

            current_sparsity_level = FLAGS.init_sparsity_level
            for prune_step in range(0, FLAGS.num_pruning_steps):
                # restore saved model
                saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
                # compute current threshold or sparsity level
                if FLAGS.pruning_threshold is None:
                    if (prune_step > 0):
                        current_sparsity_level = increase_sparsity_level(current_sparsity_level, FLAGS.max_sparsity_level, FLAGS.sparsity_increase_step)
                print('current sparsity level', current_sparsity_level)
                # print('threshold', FLAGS.pruning_threshold)

                # quantize layers
                for layer in layers_to_compress:
                    layer.prune_weights(mon_sess, FLAGS.pruning_threshold, current_sparsity_level)

                # last_c1_values = mon_sess.run(layers_to_compress[0].weights)
                # print('last c1 values', last_c1_values)

                # save current weights
                saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'pruned_model.ckpt'))
                # restore for retraining
                saver.restore(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'pruned_model.ckpt'))

                # new_values = mon_sess.run(layers_to_compress[0].weights))

                # retrain
                for step in range(0, FLAGS.num_pruning_retrain_steps):
                    # run gradients
                    eval_grad = mon_sess.run(gradients_list)
                    # print(eval_grad)
                    # prune gradients
                    eval_grad[0] = net.fc6.prune_gradients(eval_grad[0])
                    eval_grad[1] = net.fc7.prune_gradients(eval_grad[1])
                    eval_grad[2] = net.fc8.prune_gradients(eval_grad[2])

                    lr = FLAGS.learning_rate *0.95**(prune_step/100)
                    # print('learning rate', mon_sess.run(learning_rate))

                    
                    # assign weights
                    for i in range(0, len(variables_to_train)):
                        mon_sess.run(compute_weight_updates_op[i], feed_dict={grads_placeholders[i]: eval_grad[i] * lr})
                        
                    global_step_count += 1
                    mon_sess.run(global_step_assign_op, feed_dict={global_step_ph: global_step_count})

                    # compute loss
                    prune_loss, summary = mon_sess.run([total_loss, merged_summary_op])
                    accuracy, _ = mon_sess.run([acc_op, acc_update_op])

                    print('prue step', prune_step, 'prune loss', prune_loss, 'accuracy', accuracy)

                    # last_c1_values = mon_sess.run(layers_to_compress[0].weights)
                    # print('last c1 values', last_c1_values)

                # save pruned model
                if (prune_step % 10 == 0):
                    writer.add_summary(summary, mon_sess.run(global_step))
                    saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, str(current_sparsity_level) + '_pruned_model.ckpt-' + str(mon_sess.run(global_step))))
                
            ############
            # QUANTIZE #
            ############
            # for quant_step in range(0, 1):
            #     quant_start_time = time.time()
            #     # restore model model
            #     saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
                
            #     for layer in layers_to_compress:
            #         layer.quantize_weights(mon_sess, FLAGS.num_quant_clusters)
                
            #     quant_time = time.time() - quant_start_time

            #     print('DONE')

            #     # last_c1_values = mon_sess.run(layers_to_compress[0].weights)
            #     # print('last c1 values', last_c1_values)

            #     # save weights
            #     saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'quant_model.ckpt'))
            #     # restore for retraining
            #     saver.restore(mon_sess, os.path.join(FLAGS.checkpoint_dir, 'quant_model.ckpt'))

            #     # last_c1_values = mon_sess.run(layers_to_compress[0].weights)
            #     # print('last c1 values', last_c1_values)

            #     quant_step_times = []

            #     for step in range(0, FLAGS.num_quant_retrain_steps):
            #         start_time = time.time()
            #         # get gradients
            #         eval_grad = mon_sess.run(gradients_list)
            #         # quantize gradients
            #         eval_grad[0] = net.c1.quantize_gradients(eval_grad[0])
            #         eval_grad[1] = net.c2.quantize_gradients(eval_grad[1])
            #         eval_grad[2] = net.fc3.quantize_gradients(eval_grad[2])

            #         lr = FLAGS.learning_rate *0.95**(quant_step/100)

            #         for i in range(0, len(variables_to_train)):
            #             mon_sess.run(compute_weight_updates_op[i], feed_dict={grads_placeholders[i]: eval_grad[i] * 0.0003})

            #         # compute loss and accuracy
            #         quant_loss, summary = mon_sess.run([total_loss, merged_summary_op])
            #         accuracy, _ = mon_sess.run([acc_op, acc_update_op])

            #         print('quant loss', quant_loss, 'accuracy', accuracy)

            #         quant_step_times.append(time.time()- start_time) 
                    
            #         global_step_count += 1
            #         mon_sess.run(global_step_assign_op, feed_dict={global_step_ph: global_step_count})
                    

                
            #     writer.add_summary(summary, mon_sess.run(global_step))
            #     saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + '_quant_model.ckpt-' + str(mon_sess.run(global_step))))

            #     end_time = time.time()

            # for layer in layers_to_compress:
            #     layer.assign_clusters(mon_sess)
            #     np.save(os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + '_codebook_' + layer.name), mon_sess.run(layer.weights))
            
            # print('avg quant time', np.sum(quant_step_times) / FLAGS.num_quant_steps)
            # print('layer quant time', quant_time)
            # np.save(os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + '_avg_quant_time'), np.sum(quant_step_times) / FLAGS.num_quant_steps)
            # np.save(os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + '_layer_quant_time'), quant_time)


            # writer.add_summary(summary, mon_sess.run(global_step))
            # saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, str(FLAGS.num_quant_clusters) + '_quantized_model.ckpt-' + str(mon_sess.run(global_step))))


if __name__ == '__main__':
    main()
