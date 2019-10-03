import numpy as np
import tensorflow as tf
import sys
sys.path.append('./../')

from layers.conv_layer import conv2d
from layers import dense_layer


from models.mnist_quant import CNN
from models.mnist_plain import mnist_plain

from datasets.dataset_factory import get_dataset


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

tf.app.flags.DEFINE_string('checkpoint_dir', '../mnist_debug',
                           'Directory for saving and restoring checkpoints.')

#######################
# Training Flags #
#######################
tf.app.flags.DEFINE_integer(
    'num_epochs', 4,
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
    # optimizer = tf.train.AdamOptimizer(
    #     learning_rate,
    #     beta1=FLAGS.adam_beta1,
    #     beta2=FLAGS.adam_beta2,
    #     epsilon=FLAGS.opt_epsilon)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    return optimizer


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

        images = tf.math.divide(images, np.float32(255))

        images.set_shape([FLAGS.batch_size, 28, 28, 1])

        labels -= FLAGS.labels_offset

        onehot_labels = tf.one_hot(
            labels, num_classes - FLAGS.labels_offset)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        summaries.add(tf.summary.image('image', images))

        logits, predictions = mnist_plain(images)

        loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits, label_smoothing=FLAGS.label_smoothing, weights=1.0))

        # Gather initial summaries.

        total_loss = tf.losses.get_total_loss()

        summaries.add(tf.summary.scalar('loss/%s' %
                                        total_loss.op.name, total_loss))

        streaming_accuracy = tf.contrib.metrics.accuracy(labels, predictions)
        summaries.add(tf.summary.scalar('accuracy', streaming_accuracy))

        model_variables = tf.contrib.framework.get_model_variables()

        variables_to_train = tf.trainable_variables()

        learning_rate = configure_learning_rate(num_samples, global_step)
        optimizer = configure_optimizer(learning_rate)
        
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        gradient_vars = optimizer.compute_gradients(
            loss_op, variables_to_train)

        train_step = optimizer.apply_gradients(
            gradient_vars, global_step=global_step)

        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.checkpoint_dir,
                                               config=config, 
                                               save_summaries_steps=10) as mon_sess:

            for i in range(0, FLAGS.num_epochs):
                for j in range(0, num_steps):

                    _, loss = mon_sess.run([train_step, total_loss])

                    print('loss', loss)
                    print(' ')
                    print('accuracy', mon_sess.run(streaming_accuracy))
                    print(' ')

if __name__ == '__main__':
    main()
