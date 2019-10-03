from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import sys
sys.path.append('./../')

from datasets.dataset_factory import get_dataset
from nets import nets_factory
from preprocessing import preprocessing_factory

from models.vgg_16 import vgg_16

tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', 2,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '../awp_vgg_10_pruned/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', './tmp/eval_dir/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'flowers', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './../tmp/flowers', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v2_50', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  g = tf.Graph()
  profiler = tf.profiler.Profiler(g)
  with g.as_default():
    run_meta = tf.RunMetadata()
    tf_global_step = tf.train.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset, num_classes, num_samples = get_dataset(
            'mnist',
            'train',
            './../tmp/mnist')

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(num_classes - FLAGS.labels_offset),
        is_training=False)

    #####################################
    # Select the preprocessing function #
    #####################################
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            'vgg_16',
            is_training=True)

    eval_image_size = 224

    dataset = dataset.map(lambda image, label: (image_preprocessing_fn(
            image, eval_image_size, eval_image_size), label))

    dataset = dataset.batch(FLAGS.batch_size)


    #########################
    # Load from the dataset #
    #########################
    # make iterator
    # TODO change iterator type
    iterator = dataset.make_one_shot_iterator()

    [images, labels] = iterator.get_next()

    # labels -= FLAGS.labels_offset

    labels = tf.cast(labels, tf.int32)

    print('batch size', FLAGS.batch_size)

    images_var = tf.Variable(
            tf.zeros([FLAGS.batch_size, eval_image_size, eval_image_size, 1]), name='images_var')

    images_ph = tf.placeholder(
        tf.float32, [FLAGS.batch_size, eval_image_size, eval_image_size, 1], name='images_ph')

    set_images_var = images_var.assign(images_ph)

    ###################
    # Define the model #
    ###################
    net = vgg_16(images_ph, num_classes)

    logits = net.forward_pass(images_var)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          tf.contrib.framework.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = tf.contrib.framework.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({
        'Accuracy': tf.contrib.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': tf.contrib.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    saver = tf.train.Saver()

    session = tf.Session()
    with session:
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        session.run(init)
        saver.restore(session, checkpoint_path)
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        for i in range(0, 1):
            preds = session.run(fetches=predictions, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta)
            profiler.add_step(i, run_meta)
            print(preds)

    flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('total flops', flops.total_float_ops)

if __name__ == '__main__':
  tf.app.run()
