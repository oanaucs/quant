import numpy as np
import tensorflow as tf
import sys
sys.path.append('./../')

from models.vgg_16 import vgg_16


from preprocessing import preprocessing_factory
from datasets.dataset_factory import get_dataset

from load_pretrained_ckpt.inspect_ckpt import load_variables


#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'flowers', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './../tmp/mnist', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 
    # '../slim_pretrained_ckpt/vgg_16.ckpt',
    None,
    'Path to pretrained model for warm start training. If None, cold start.')

tf.app.flags.DEFINE_string('checkpoint_dir', '../vgg_mnist/',
                           'Directory for saving and restoring checkpoints.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

#######################
# Training Flags #
#######################
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

tf.app.flags.DEFINE_float('learning_rate', 0.0003, 'Initial learning rate.')

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
        config.gpu_options.per_process_gpu_memory_fraction = 0.6

        # Create global_step
        global_step = tf.train.create_global_step()

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
            'mnist',
            'train',
            './../tmp/mnist')

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

        tf.summary.tensor_summary('ground truth', labels)


        labels -= FLAGS.labels_offset

        labels = tf.one_hot(
            labels, num_classes - FLAGS.labels_offset, dtype=tf.int32)

        tf.summary.image('train image', images)

        images_var = tf.Variable(
            tf.zeros([FLAGS.batch_size, train_image_size, train_image_size, 1]), 
            trainable=False, name='images_var')

        images_ph = tf.placeholder(
            tf.float32, [FLAGS.batch_size, train_image_size, train_image_size, 1], name='images_ph')

        set_images_var = images_var.assign(images_ph)

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
        net = vgg_16(images_ph, num_classes, 
            ckpt_var_dict=variables_to_restore, 
            pruning_threshold=0.05)

        logits, predictions = net.train(images_var)

        tf.summary.tensor_summary('predictions', predictions)

        predictions = tf.one_hot(
            predictions, num_classes - FLAGS.labels_offset, dtype=tf.int32)

        layers_to_compress = net.layers_as_list()

        loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits, label_smoothing=FLAGS.label_smoothing, weights=1.0))

        ######################
        # Summarize progress #
        ######################

        # # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        total_loss = tf.losses.get_total_loss()

        summaries.add(tf.summary.scalar('loss/%s' %
                                        total_loss.op.name, total_loss))

        streaming_accuracy = tf.contrib.metrics.accuracy(predictions=predictions, 
            labels=labels)
        summaries.add(tf.summary.scalar('accuracy', streaming_accuracy))

        for layer in layers_to_compress:
            summaries.add(tf.summary.histogram('activations/' + layer.name, layer.weights))
            summaries.add(tf.summary.scalar('sparsity/' + layer.name,
                tf.nn.zero_fraction(layer.weights)))

        for variable in tf.contrib.framework.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        model_variables = tf.contrib.framework.get_model_variables()

        variables_to_train = tf.trainable_variables()

        # numerical check
        
        # remove
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     FLAGS.moving_average_decay, tf.train.get_global_step())

        learning_rate = configure_learning_rate(num_samples, tf.train.get_global_step())
        optimizer = configure_optimizer(learning_rate)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        gradients_vars = optimizer.compute_gradients(
            total_loss, tf.trainable_variables())

        gradients = [grad for grad, var in gradients_vars]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        
        with tf.control_dependencies(update_ops):
            train_step = optimizer.apply_gradients(
            gradients_vars, global_step=tf.train.get_global_step())
        
        # if FLAGS.moving_average_decay:
        #     with tf.control_dependencies([train_step]):
        #         # Update ops executed locally by trainer.
        #         update_ops.append(variable_averages.apply(model_variables))


        ######################
        # Start the training #
        ######################
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.checkpoint_dir,
                                               save_summaries_steps=2,
                                               config=config) as mon_sess:
            if restore_model_op:
                restore_model_op(mon_sess)

            for i in range(0, 1):
                eval_images = mon_sess.run(images)
                eval_labels = mon_sess.run(labels)

                print('images', np.count_nonzero(eval_images))

                print('image size', eval_images.shape)

                _, loss = mon_sess.run([train_step, total_loss], feed_dict={
                    images_var: eval_images, labels: eval_labels}, 
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                  run_metadata=run_meta)

                print('loss', loss)
                print('gradients', np.count_nonzero(mon_sess.run(gradients[0])))
                
            # ###########
            # # PRUNE #
            # # print('pruning')
            # # print('layers to compress', layers_to_compress)
            # for i in range(0, 50):
            #     print('current step', i)
            #     # for as much memory as we have... omit fc layers
            #     for j in range(0, len(layers_to_compress)):
            #         layers_to_compress[j].prune_weights(mon_sess)

            #     # retrain
            #     eval_images = mon_sess.run(images)
            #     eval_labels = mon_sess.run(labels)
            #     print(eval_labels)

            #     gradient_values, prune_loss = mon_sess.run(
            #         [gradients, total_loss], feed_dict={images_var: eval_images, labels: eval_labels})

            #     # print('grad values', gradient_values)

            #     print('prune loss', prune_loss)

            #     layers_to_compress_names = []
            #     for layer in layers_to_compress:
            #         layers_to_compress_names.append(layer.name)

            #     # print('gradient', gradients[i].name)

            #     print('layers', layers_to_compress_names)

            #     pruned_gradients = {}
            #     for i in range(len(gradients)):
            #         for j in range(0, len(layers_to_compress_names)):
            #             if (gradients[i].name.find(layers_to_compress_names[j]) != -1):
            #                 pruned_gradients[gradients[i]] = layers_to_compress[j].prune_gradients(
            #                     gradient_values[i])
            #             else:
            #                 pruned_gradients[gradients[i]] = gradient_values[i]


            #     mon_sess.run(train_step, feed_dict=pruned_gradients)

            # last_c1_values = mon_sess.run(layers_to_compress[10].weights)
            # # print('last prune loss', prune_loss)
            # # print('prune c1 values', last_c1_values)
            # ###########
            # # QUANTIZE #
            # for i in range(0, 50):
            #     for layer in layers_to_compress:
            #         layer.quantize_weights(mon_sess)

            #     # retrain
            #     eval_images = mon_sess.run(images)
            #     eval_labels = mon_sess.run(labels)

            #     gradient_values, quantization_loss = mon_sess.run(
            #         [gradients, total_loss], feed_dict={images_var: eval_images, labels: eval_labels})

            #     print('quantization loss', quantization_loss)
            #     print('c1 values')
            #     c1_values = mon_sess.run(layers_to_compress[10].weights)
            #     print(c1_values)

            #     print('difference')
            #     print(last_c1_values - c1_values)

            #     quantized_gradients = {}
            #     for i in range(len(gradients)):
            #         for j in range(0, len(layers_to_compress_names)):
            #             if (gradients[i].name.find(layers_to_compress_names[j]) != -1):
            #                 pruned_gradients[gradients[i]] = layers_to_compress[j].quantize_gradients(
            #                     gradient_values[i])
            #             else:
            #                 pruned_gradients[gradients[i]] = gradient_values[i]

            #     mon_sess.run(train_step, feed_dict=quantized_gradients)


if __name__ == '__main__':
    main()
