# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for building the input for the prediction model.

"""

import os
import re
import tensorflow as tf
from training_flags import FLAGS
from .base_data_reader import BaseDataReader


class BairDataReader(BaseDataReader):

    def __init__(self,
                 dataset_dir=None,
                 train_val_split=None,
                 *args,
                 **kwargs):
        """
        Dataset class for the BAIR and Google Push datasets.
        :param dataset_dir: (str, optional) path to dataset directory, containing the /train and a /test directories.
                            Defaults to FLAGS.bair_dir or FLAGS.google_dir, defined in training_flags.py, depending on
                            the dataset_name parameter.
        """
        super(BairDataReader, self).__init__(*args, **kwargs)
        self.dataset_name = 'bair'
        self.COLOR_CHAN = 3
        self.IMG_WIDTH = 64
        self.IMG_HEIGHT = 64
        self.STATE_DIM = 3
        self.ACTION_DIM = 4
        self.ORIGINAL_WIDTH = 64
        self.ORIGINAL_HEIGHT = 64
        self.data_dir = dataset_dir if dataset_dir else FLAGS.bair_dir
        self.train_val_split = train_val_split if train_val_split else FLAGS.train_val_split
        self.train_filenames, self.val_filenames, self.test_filenames = self.set_filenames()

    def _parse_sequences(self, serialized_example):

        image_seq, state_seq, action_seq = [], [], []

        for i in range(self.sequence_length_to_use):

            image_name = str(i) + '/image_aux1/encoded'
            action_name = str(i) + '/action'
            state_name = str(i) + '/endeffector_pos'
            # double_view option (check the colab repo)

            features = {image_name: tf.FixedLenFeature([1], tf.string),
                        action_name: tf.FixedLenFeature([self.ACTION_DIM], tf.float32),
                        state_name: tf.FixedLenFeature([self.STATE_DIM], tf.float32)}

            features = tf.parse_single_example(serialized_example, features=features)

            image = tf.decode_raw(features[image_name], tf.uint8)
            image = tf.reshape(image, shape=[1, self.ORIGINAL_HEIGHT * self.ORIGINAL_WIDTH * self.COLOR_CHAN])
            image = tf.reshape(image, shape=[self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH, self.COLOR_CHAN])

            assert self.IMG_HEIGHT == self.IMG_WIDTH, 'Unequal height and width unsupported'

            # Make the image square e.g.: 640x512 ==> 512x512
            crop_size = min(self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH)
            image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)

            # add a forth dimension to later stack with sequence len
            image = tf.reshape(image, [1, crop_size, crop_size, self.COLOR_CHAN])

            # Resize the image to 64x64 using bicubic inteporlation (downgrades the resolution)
            image = tf.image.resize_bicubic(image, [self.IMG_HEIGHT, self.IMG_WIDTH])

            # normalizes to [0,1] range
            image = tf.cast(image, tf.float32) / 255.0

            # add the new frame to a list with a sequence of frames. shape: seq_len*(1, 64, 64, 3)
            image_seq.append(image)

            if FLAGS.use_state:
                state = tf.reshape(features[state_name], shape=[1, self.STATE_DIM])
                state_seq.append(state)

                action = tf.reshape(features[action_name], shape=[1, self.ACTION_DIM])
                action_seq.append(action)

        # stack the list of frames in a single tensor. shape: (seq_len, 64, 64, 3)
        image_seq = tf.concat(image_seq, 0)

        if FLAGS.use_state:
            state_seq = tf.concat(state_seq, 0)
            action_seq = tf.concat(action_seq, 0)

            states_t = state_seq[:-1, :]
            states_tp1 = state_seq[1:, :]
            delta_xy = states_tp1[:, :2] - states_t[:, :2]
            return {'images': image_seq,
                    'actions': action_seq,
                    'states': state_seq,
                    'action_targets': delta_xy}
        else:
            zeros_action = tf.zeros([self.sequence_length_to_use, self.ACTION_DIM])
            zeros_state = tf.zeros([self.sequence_length_to_use, self.STATE_DIM])
            zeros_targets = tf.zeros([self.sequence_length_to_use-1, 2])
            return {'images': image_seq,
                    'actions': zeros_action,
                    'states': zeros_state,
                    'action_targets': zeros_targets}

    def num_examples_per_epoch(self, mode):
        """
        SOURCE:
        https://github.com/alexlee-gk/video_prediction/blob/master/video_prediction/datasets/softmotion_dataset.py
        """
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        if mode == 'train':
            filenames = self.train_filenames
        elif mode == 'val':
            filenames = self.val_filenames
        elif mode == 'test':
            filenames = self.test_filenames

        for filename in filenames:
            match = re.search('traj_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count
