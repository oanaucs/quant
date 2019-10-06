from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import time

import numpy as np

import sys
sys.path.append('./../../')
sys.path.append('./../')

import os

from tensorflow.python.framework import graph_util

from datasets.dataset_factory import get_dataset
from preprocessing import preprocessing_factory

from models.mnist_quant import CNN

#################
# Dataset Flags #
#################

tf.app.flags.DEFINE_string(
    'dataset_name', 'mnist', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './../../tmp/mnist', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'test_image_size', 28, 'Train image size')

##############
# Test Flags #
##############

tf.app.flags.DEFINE_string('checkpoint_path', '/media/oanaucs/Data/awp_trained_models/mnist_prune_0.01/0.035_pruned_model.ckpt-23071',
                           'Directory for saving and restoring checkpoints.')

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'eval_dir', './tmp/eval_dir/', 'Directory where the results are saved to.')

# tf.app.flags.DEFINE_integer(
#     'num_preprocessing_threads', 4,
#     'The number of threads used to create the batches.')


FLAGS = tf.app.flags.FLAGS

def count_non_zero(ckpt_path):
    reader = tf.train.NewCheckpointReader(ckpt_path)
    param_map = reader.get_variable_to_shape_map()

    non_zero_count = 0

    for key in param_map:
        if 'Adam' not in key:
            print("tensor_name: ", key)
            tensor = reader.get_tensor(key)
            non_zero_count += np.count_nonzero(tensor)

    print('non zero', non_zero_count)


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  g = tf.Graph()
  profiler = tf.profiler.Profiler(g)
  with g.as_default():
    run_meta = tf.RunMetadata()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    

    #####################################
    # Select the preprocessing function #
    #####################################
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        'lenet',
        is_training=False)

    ######################
    # Select the dataset #
    ######################
    dataset, num_classes, num_samples = get_dataset(
            FLAGS.dataset_name,
            FLAGS.dataset_split_name,
            FLAGS.dataset_dir)

    dataset = dataset.map(lambda image, label: (image_preprocessing_fn(
            image, FLAGS.test_image_size, FLAGS.test_image_size), label))

    dataset = dataset.batch(FLAGS.batch_size)
    
    #########################
    # Load from the dataset #
    #########################
    # make iterator
    iterator = dataset.make_one_shot_iterator()

    [images, labels] = iterator.get_next()

    labels = tf.cast(labels, tf.int32)

    # define number of eval steps
    if FLAGS.max_num_batches is not None:
        num_batches = FLAGS.max_num_batches
    else:
        # This ensures that we make a single pass over all of the data.
        num_batches = int(num_samples / float(FLAGS.batch_size))

    ####################
    # Create the model #
    ####################
    net = CNN(FLAGS.batch_size, num_classes)
    logits, predictions = net.forward_pass(images)
    
    #############
    # Summarize #
    #############
    
    # Define the metrics:
    acc_op, acc_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    ########################
    # Create saver and sess#
    ########################
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        model_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % model_path)
    
    saver = tf.train.Saver()

    model_variables = tf,tf.trainable_variables()
    # model_saver = tf.train.Saver()

    session = tf.Session()
    
    ############
    # Evaluate #
    ###########
    step_times = []
    with session.as_default():
        # init variables
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        session.run(init)
        # restore model
        # saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
        saver.restore(session, model_path)
        # saver.save(session, os.path.join('/media/oanaucs/Data/awp_trained_models/mnist_q/', str(FLAGS.num_quant_clusters) + '_weights_model.ckpt-' + str(mon_sess.run(global_step))))

        
        # run forward pass for eval steps
        try:
            for i in range(0, num_batches):
                start = time.time()
                preds = session.run(fetches=predictions, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta)
                end = time.time() - start
                profiler.add_step(i, run_meta)
                step_times.append(end)
                accuracy, _ = session.run([acc_op, acc_update_op])
                print('step', i, 'accuracy', accuracy)
        except tf.errors.OutOfRangeError:
            pass
        
        # # write to pb
        # output_graph_def = graph_util.convert_variables_to_constants(
        #     session, g.as_graph_def(), 
        #     ['Cast_5'] # unrelated nodes will be discarded
        # ) 

        # output_graph = os.path.join('/media/oanaucs/Data/awp_trained_models/mnist_sparsity/frozen/', str(0.1) + 'frozen_graph.pb')

        # with tf.gfile.GFile(output_graph, "wb") as f:
        #     f.write(output_graph_def.SerializeToString())
        # print("%d ops in the final graph." % len(output_graph_def.node))

    # flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
    # params = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    # print('total flops', flops.total_float_ops)
    # # print('params', params)

    # print('avg step time', np.sum(step_times)/num_batches)

    count_non_zero(model_path)

    # print('avgquant', np.load('/media/oanaucs/Data/awp_trained_models/mnist_q/256_avg_quant_time.npy'))
    # print('layer quant', np.load('/media/oanaucs/Data/awp_trained_models/mnist_q/256_layer_quant_time.npy'))




if __name__ == '__main__':
  tf.app.run()