# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'flowers_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 3320, 'validation': 350}

_NUM_CLASSES = 5

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}


def _parse_(serialized_example):
    # feature = {'image_raw':tf.FixedLenFeature([], tf.string),
    #             'label':tf.FixedLenFeature([], tf.int64)}

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    example = tf.parse_single_example(serialized_example, keys_to_features)
    # image = tf.image.decode_raw(example['image_raw'], tf.int64)
    # remember to parse in int64. float will raise error
    image = tf.image.decode_png(example['image/encoded'], channels=3)
    image.set_shape([32, 32, 3])
    image = tf.cast(image, tf.float32)
    # tf.reshape(image, [32, 32, 3])
    label = tf.cast(example['image/class/label'], tf.int32)
    return (image, label)


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading flowers.

    Args:
      split_name: A train/validation split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/validation split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        files = os.listdir(dataset_dir)
        file_pattern = _FILE_PATTERN
        file_pattern = [os.path.join(dataset_dir, f)
                        for f in files if (split_name in f)]

    return (tf.data.TFRecordDataset(file_pattern).map(lambda x: _parse_(x)), _NUM_CLASSES, SPLITS_TO_SIZES[split_name])
