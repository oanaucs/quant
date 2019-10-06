from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import numpy as np

import sys
sys.path.append('./../../')
sys.path.append('./../')

import os

from datasets.dataset_factory import get_dataset
from preprocessing import preprocessing_factory

from models.alexnet import alexnet

#################
# Dataset Flags #
#################

tf.app.flags.DEFINE_string(
    'dataset_name', 'fashion_mnist', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '.\\..\\..\\tmp\\fashion_mnist', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'test_image_size', 227, 'Train image size')

##############
# Test Flags #
##############

tf.app.flags.DEFINE_string('checkpoint_path', '.\\..\\..\\trained_models\\alexnet_fmnist',
                           'Directory for saving and restoring checkpoints.')

tf.app.flags.DEFINE_integer(
    'batch_size', 128, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'eval_dir', './tmp/eval_dir/', 'Directory where the results are saved to.')

# tf.app.flags.DEFINE_integer(
#     'num_preprocessing_threads', 4,
#     'The number of threads used to create the batches.')


FLAGS = tf.app.flags.FLAGS


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
    net = alexnet(FLAGS.batch_size, num_classes, 1)
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

    session = tf.Session()
    
    ############
    # Evaluate #
    ###########
    with session.as_default():
        # init variables
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        session.run(init)
        # restore model
        saver.restore(session, model_path)
        
        #run forward pass for eval steps
        try:
            for i in range(0, num_batches):
                preds = session.run(fetches=predictions, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta)
                # profiler.add_step(i, run_meta)
                accuracy, _ = session.run([acc_op, acc_update_op])
                print('step', i, 'accuracy', accuracy)
        except tf.errors.OutOfRangeError:
            pass

    # flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
    # print('total flops', flops.total_float_ops)


if __name__ == '__main__':
  tf.app.run()