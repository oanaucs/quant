import numpy as np
import tensorflow as tf
import sys
import os 

sys.path.append('./../../')
sys.path.append('./../')

from tensorflow.python.client import timeline

from layers.conv_layer import conv2d
from layers import dense_layer

from models.fmnist_quant import FCNN

from datasets.dataset_factory import get_dataset
from preprocessing import preprocessing_factory

from utils.count_params import count_nonzero_weights

from utils.prune_utils import prune_one_step, increase_sparsity_level
from utils.quant_utils import quantize_one_step, assign_clusters, export_dict_to_file

#################
# Dataset Flags #
#################

tf.app.flags.DEFINE_string(
    'dataset_name', 'mnist', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './../../tmp/mnist', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 28, 'Train image size')

##################
# Training Flags #
##################
tf.app.flags.DEFINE_string('checkpoint_dir', '../../trained_models/fmnist',
                           'Directory for saving and restoring checkpoints.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_epochs', 50,
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

tf.app.flags.DEFINE_float('learning_rate', 0.003, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0003,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 25,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#################
# Pruning Flags #
#################

tf.app.flags.DEFINE_integer(
    'num_pruning_steps', 10,
    'Number of pruning steps.')

tf.app.flags.DEFINE_integer(
    'num_pruning_retrain_steps', 10,
    'Number of retrain steps after_pruning.')

tf.app.flags.DEFINE_integer(
    'pruning_threshold', None,
    'Pruning threshold. If set to None, adaptive sparsity level will be used.')

tf.app.flags.DEFINE_float(
    'init_sparsity_level', 0.3,
    'Initial sparsity level.')

tf.app.flags.DEFINE_float(
    'max_sparsity_level', 0.5,
    'Maximum sparsity level. Depending on the number of steps can be achieved or not')

tf.app.flags.DEFINE_float(
    'sparsity_increase_step', 0.05,
    'Step for increasing the sparsity level after one pruning step.')

######################
# Quantization Flags #
######################

tf.app.flags.DEFINE_integer(
    'num_quant_retrain_steps', 100,
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


def configure_optimizer(learning_rate):
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    return optimizer


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    g = tf.Graph()
    # profiler = tf.profiler.Profiler(g)
    with g.as_default():
        run_meta = tf.RunMetadata()
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        # Create global_step
        global_step = tf.train.create_global_step()

        summary_writer = tf.contrib.summary.create_file_writer(FLAGS.checkpoint_dir, flush_millis=10000)

        ######################
        # Select preprocessing #
        ######################

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

        # more epochs for retraining 
        dataset = dataset.repeat(FLAGS.num_epochs).shuffle(True).batch(FLAGS.batch_size)

        #########################
        # Load from the dataset #
        #########################
        # make iterator
        iterator = dataset.make_one_shot_iterator()

        [images, labels] = iterator.get_next()

        images = tf.identity(images, name='images')
        
        # create labels
        onehot_labels = tf.one_hot(labels, num_classes)
        
        # add images to summary
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.image('image', images))
        
        ##################
        # Create network #
        ##################
        quant_net = FCNN(FLAGS.batch_size, num_classes)
        logits, predictions = quant_net.train(images)
        
        # get a list of layers to compress
        layers_to_compress = quant_net.layers_as_list()

        # create loss
        loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits, label_smoothing=FLAGS.label_smoothing, weights=1.0))

        total_loss = tf.losses.get_total_loss()

        summaries.add(tf.summary.scalar('loss/%s' %
                                        total_loss.op.name, total_loss))

        # gather summaries
        streaming_accuracy, acc_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions, name='streaming_accuracy')
        summaries.add(tf.summary.scalar('accuracy', streaming_accuracy))
    
        for layer in layers_to_compress:
            summaries.add(tf.summary.histogram('activations/' + layer.name, layer.weights))
            summaries.add(tf.summary.scalar('sparsity/' + layer.name,
                tf.nn.zero_fraction(layer.weights)))

        #######################
        # Define backprop ops #
        #######################
        # define trainable variables
        variables_to_train = tf.trainable_variables()

        # define learning rate
        learning_rate = configure_learning_rate(num_samples, global_step)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
        
        # define optimizer and backprop operations
        optimizer = configure_optimizer(learning_rate)

        gradient_vars = optimizer.compute_gradients(
            loss_op, variables_to_train)

        gradients = [grad for grad, var in gradient_vars]

        train_step = optimizer.apply_gradients(
            gradient_vars, global_step=global_step)

        # merge summaries
        merged_summary_op = tf.summary.merge_all()

        ################
        # Create saver #
        ################
        saver = tf.train.Saver()
        model_variables_saver = tf.train.Saver(tf.trainable_variables())

        writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, tf.get_default_graph())

        ##############################
        # Define additional variables#
        ##############################
        avg_times = dict()
        nonzero_weights = dict()
        
        sparsity_opts = dict()
        sparsity_opts['init_sparsity_level'] = FLAGS.init_sparsity_level
        sparsity_opts['max_sparsity_level'] = FLAGS.max_sparsity_level
        sparsity_opts['sparsity_increase_step'] = FLAGS.sparsity_increase_step


        ##################
        # Start training #
        ##################
        with tf.Session() as mon_sess:
            # init variables
            mon_sess.run(tf.global_variables_initializer())
            mon_sess.run(tf.local_variables_initializer())

            # restore from checkpoint if available
            last_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if last_ckpt:
                saver.restore(mon_sess, last_ckpt)

            ####################
            # INITIAL TRAINING #
            ####################
            # for i in range(0, FLAGS.num_epochs):
            #     for j in range(0, num_steps):

            #         _, loss, summary, _ = mon_sess.run([train_step, loss_op, merged_summary_op, acc_update_op])
            #         print('training loss', loss)
                    
            #         if (j % 100 == 0):
            #             writer.add_summary(summary, mon_sess.run(global_step))
            #             save_path = saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, "model.ckpt-" + str(mon_sess.run(global_step))))
            
            # writer.add_summary(summary, mon_sess.run(global_step))
            # save_path = saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, "initial_model.ckpt-" + str(mon_sess.run(global_step))))

            # #########
            # # PRUNE #
            #########
            current_sparsity_level = sparsity_opts['init_sparsity_level']
            pruning_times = []

            for prune_step in range(0, FLAGS.num_pruning_steps):
                print('pruning step', prune_step)
                if FLAGS.pruning_threshold is None:
                    current_sparsity_level = increase_sparsity_level(current_sparsity_level, sparsity_opts)
                    print('current sparsity level', current_sparsity_level)

                step_time = prune_one_step(mon_sess,
                                            layers_to_compress, 
                                            gradients, 
                                            train_step, 
                                            loss_op,
                                            FLAGS.pruning_threshold,
                                            current_sparsity_level)
                pruning_times.append(step_time)

                # retrain for 1 epoch
                for prune_retrain_step in range(0, FLAGS.num_pruning_retrain_steps):
                    _, loss, summary = mon_sess.run([train_step, loss_op, merged_summary_op])
                        #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                        #run_metadata=run_meta)
                        
                    print('pruning loss', loss)
                    
                    # profiler.add_step(tf.train.global_step(session, global_step), run_meta)

                if (prune_step % 10):
                    writer.add_summary(summary, mon_sess.run(global_step))
                    save_path = saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, str(current_sparsity_level) + "pruned_model.ckpt-" + str(mon_sess.run(global_step))))
                    nonzero_weights[current_sparsity_level] = count_nonzero_weights(save_path)
                    print('nonzero weights', current_sparsity_level, nonzero_weights[current_sparsity_level])

            avg_times['pruning'] = np.sum(pruning_times) / FLAGS.num_pruning_steps

            ###########

            ############
            ## # QUANTIZE #
            #quant_time = quantize(mon_sess, layers_to_compress, gradients, learning_rate, train_step, loss_op)
            
            #avg_times['quant'] = quant_time

            #for quant_retrain_step in range(0, num_steps):
                #_, loss, summary = mon_sess.run([train_step, loss_op, merged_summary_op],
                    #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    #run_metadata=run_meta)
                ## profiler.add_step(tf.train.global_step(mon_sess, global_step), run_meta)
        
            #writer.add_summary(summary, mon_sess.run(global_step))    
            #save_path = saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, "model.ckpt-" + str(mon_sess.run(global_step))))
            #print('saving checkpoint at', save_path)

                
            #assign_clusters(mon_sess, FLAGS.checkpoint_dir, layers_to_compress)
            #writer.add_summary(summary, mon_sess.run(global_step))    
            #save_path = model_variables_saver.save(mon_sess, os.path.join(FLAGS.checkpoint_dir, "quantized_model.ckpt-" + str(mon_sess.run(global_step))))

            #print('saving final checkpoint at', save_path)


            # # ###########
            # # # PROFILE #
            # option_builder = tf.profiler.ProfileOptionBuilder
            # opts = (option_builder(option_builder.time_and_memory()).
            #         with_step(-1). # with -1, should compute the average of all registered steps.
            #         # with_file_output('test-%s.txt' % FLAGS.out).
            #         select(['micros','bytes','occurrence']).order_by('micros').
            #         build())
            # # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
            # profiler.profile_operations(options=opts)


            # fetched_timeline = timeline.Timeline(run_meta.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open(os.path.join(FLAGS.checkpoint_dir, 'timeline.json'), 'w') as f:
            #     f.write(chrome_trace)


        # additional information
        export_dict_to_file(FLAGS, FLAGS.checkpoint_dir, 'train_opts')
        export_dict_to_file(avg_times, FLAGS.checkpoint_dir, 'avg_time')
        export_dict_to_file(nonzero_weights, FLAGS.checkpoint_dir, 'nonzero_count')
            

            
if __name__ == '__main__':
    main()
