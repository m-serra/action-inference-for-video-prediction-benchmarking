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

import os, sys, re
import random
import multiprocessing
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from training_flags import FLAGS


class DataReader(object):

    def __init__(self,
                 dataset_name=None,
                 dataset_dir=None,
                 shuffle=False,
                 dataset_repeat=None,
                 sequence_length_train=None,
                 sequence_length_test=None):
        """
        Dataset class for the BAIR and Google Push datasets.
        :param data: (str, optional) Name of the dataset. One of 'bair', 'googlepush', 'bair_predictions',
                     'googlepush_predictions'
        :param dataset_dir: (str, optional) path to dataset directory, containing the /train and a /test directories.
                            Defaults to FLAGS.bair_dir or FLAGS.google_dir, defined in training_flags.py, depending on
                            the dataset_name parameter.
        :param shuffle: (boolean, optional) whether to shuffle the train/val filenames and tfrecord samples. Defaults
                        to FLAGS.shuffle.
        :param dataset_repeat: (int, optional) number of times the dataset can be iterated. Default allows indefinite
                               iteration.
        :param sequence_length_train: (int, optional) number of timesteps to use for training and validation
        :param sequence_length_test: (int, optional) number of timesteps to use for test
        """
        self.COLOR_CHAN = 3
        self.IMG_WIDTH = 64
        self.IMG_HEIGHT = 64
        self.filenames = None
        self.dataset_name = FLAGS.dataset.lower() if dataset_name is None else dataset_name.lower()
        self.dataset_repeat = FLAGS.n_epochs if dataset_repeat is None else dataset_repeat
        self.shuffle = FLAGS.shuffle if shuffle is None else shuffle
        self.sequence_length_train = FLAGS.sequence_length_train if sequence_length_train is None else sequence_length_train
        self.sequence_length_test = FLAGS.sequence_length_test if sequence_length_test is None else sequence_length_test
        self.sequence_length = None
        self.n_threads = multiprocessing.cpu_count()

        assert self.dataset_name in ['bair', 'googlepush', 'bair_predictions', 'googlepush_predictions'], \
            "dataset must one of 'bair' 'googlepush', 'bair_predictions' or 'googlepush_predictions'."

        parse_fn_mappings = {'bair': self._parse_sequences,
                             'googlepush': self._parse_sequences,
                             'bair_predictions': self._parse_prediction_sequences,
                             'googlepush_predictions': self._parse_prediction_sequences}

        self.parse_fn = parse_fn_mappings.get(self.dataset_name)

        if self.dataset_name in ['googlepush', 'googlepush_predictions']:
            self.STATE_DIM = 5
            self.ACTION_DIM = 5
            self.ORIGINAL_WIDTH = 640
            self.ORIGINAL_HEIGHT = 512
            self.data_dir = dataset_dir if dataset_dir else FLAGS.google_dir

        elif self.dataset_name in ['bair', 'bair_predictions']:
            self.STATE_DIM = 3
            self.ACTION_DIM = 4
            self.ORIGINAL_WIDTH = 64
            self.ORIGINAL_HEIGHT = 64
            self.data_dir = dataset_dir if dataset_dir else FLAGS.bair_dir

        train_val_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        try:
            self.train_val_filenames = gfile.Glob(os.path.join(train_val_dir, '*'))
            idx_train_val_split = int(np.floor(FLAGS.train_val_split * len(self.train_val_filenames)))
            self.train_filenames = self.train_val_filenames[:idx_train_val_split]
            self.val_filenames = self.train_val_filenames[idx_train_val_split:]
        except:
            self.train_filenames = None
            self.train_filenames = None
            print("No train/val data files found.")

        try:
            self.test_filenames = gfile.Glob(os.path.join(test_dir, '*'))
        except:
            self.test_filenames = None
            print("No test data files found.")

    def build_tfrecord_iterator(self, mode='train'):
        """Create input tfrecord iterator
        Args:
        -----
            dataset: the dataset to be used. Either 'BAIR' or 'GooglePush'
            training: use training data files (if True) or use validation data files (if False)

        Returns:
        --------
            An iterator that return images, actions, states, distances, angles
        """
        assert mode in ['train', 'val', 'test'], 'Mode must be one of "train", "val" or "test"'

        # 1 - Get the names of the .tfrecord files to be read
        if mode == 'train' and self.train_filenames is not None:
            self.filenames = self.train_filenames
            self.sequence_length = self.sequence_length_train
        elif mode == 'val' and self.val_filenames is not None:
            self.filenames = self.val_filenames
            self.sequence_length = self.sequence_length_train
        elif self.test_filenames:
            self.filenames = self.test_filenames
            self.sequence_length = self.sequence_length_test

        if self.filenames is None:
            print('No files found')
            sys.exit(0)

        filename_queue = tf.data.Dataset.from_tensor_slices(self.filenames)
        if self.shuffle and mode != 'test':
           filename_queue = filename_queue.shuffle(buffer_size=len(self.filenames))

        dataset = tf.data.TFRecordDataset(filename_queue)

        if self.shuffle and mode is not 'test':
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(32 * FLAGS.batch_size, count=self.dataset_repeat))
        else:
            dataset = dataset.repeat(count=self.dataset_repeat)  # to allow multiple epochs with one_shot_iterator

        dataset = dataset.map(lambda x: self.parse_fn(x))
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

        iterator = dataset.make_one_shot_iterator()

        return iterator

    def _parse_sequences(self, serialized_example):

        image_seq, state_seq, action_seq = [], [], []

        for i in range(self.sequence_length):

            if self.dataset_name in ['googlepush']:
                image_name = 'move/' + str(i) + '/image/encoded'
                action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
                state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'

            elif self.dataset_name in ['bair']:
                image_name = str(i) + '/image_aux1/encoded'
                action_name = str(i) + '/action'
                state_name = str(i) + '/endeffector_pos'
                # double_view option (check the colab repo)

            features = {image_name: tf.FixedLenFeature([1], tf.string),
                        action_name: tf.FixedLenFeature([self.ACTION_DIM], tf.float32),
                        state_name: tf.FixedLenFeature([self.STATE_DIM], tf.float32)}

            features = tf.parse_single_example(serialized_example, features=features)

            if self.dataset_name in ['googlepush', 'googlepush_predictions']:
                image_buffer = tf.reshape(features[image_name], shape=[])
                image = tf.image.decode_jpeg(image_buffer, channels=self.COLOR_CHAN)
                image.set_shape([self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH, self.COLOR_CHAN])

            elif self.dataset_name in ['bair', 'bair_predictions']:
                image = tf.decode_raw(features[image_name], tf.uint8)
                image = tf.reshape(image, shape=[1, self.ORIGINAL_HEIGHT * self.ORIGINAL_WIDTH * self.COLOR_CHAN])
                image = tf.reshape(image, shape=[self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH, self.COLOR_CHAN])

            assert self.IMG_HEIGHT == self.IMG_WIDTH, 'Unequal height and width unsupported'

            # Make the image square 640x512 ==> 512x512
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
                    'deltas': delta_xy}
        else:
            zeros_action = tf.zeros([self.sequence_length, self.ACTION_DIM])
            zeros_state = tf.zeros([self.sequence_length, self.STATE_DIM])
            zeros_deltas = tf.zeros([self.sequence_length - 1, 2])
            return {'images': image_seq,
                    'actions': zeros_action,
                    'states': zeros_state,
                    'deltas': zeros_deltas}

    def _parse_prediction_sequences(self, serialized_example):
        image_seq, state_seq = [], []
        for i in range(self.sequence_length):
            image_name = 'move/' + str(i) + '/image/encoded'
            state_name = 'move/' + str(i) + '/state'

            if self.dataset_name in ['googlepush_predictions']:
                image_name = 'move/' + str(i) + '/image/encoded'
                state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'

            elif self.dataset_name in ['bair', 'bair_predictions']:
                image_name = str(i) + '/image_aux1/encoded'
                state_name = str(i) + '/endeffector_pos'

            features = {image_name: tf.FixedLenFeature([1], tf.string),
                        state_name: tf.FixedLenFeature([self.STATE_DIM], tf.float32)}
            features = tf.parse_single_example(serialized_example, features=features)

            image = tf.decode_raw(features[image_name], float)
            image = tf.reshape(image, shape=[1, 64 * 64 * 3])
            image = tf.reshape(image, shape=[64, 64, 3])
            assert self.IMG_HEIGHT == self.IMG_WIDTH, 'Unequal height and width unsupported'

            crop_size = min(self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH)
            image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
            image = tf.reshape(image, [1, crop_size, crop_size, self.COLOR_CHAN])
            image = tf.image.resize_bicubic(image, [self.IMG_HEIGHT, self.IMG_WIDTH])
            # image = tf.cast(image, tf.float32) / 255.0
            image_seq.append(image)

            state = tf.reshape(features[state_name], shape=[1, self.STATE_DIM])
            state_seq.append(state)

        image_seq = tf.concat(image_seq, 0)

        state_seq = tf.concat(state_seq, 0)
        states_t = state_seq[:-1, :]
        states_tp1 = state_seq[1:, :]
        delta_xy = states_tp1[:, :2] - states_t[:, :2]

        return {'images': image_seq, 'deltas': delta_xy}

    def num_examples_per_epoch(self):
        """
        SOURCE:
        https://github.com/alexlee-gk/video_prediction/blob/master/video_prediction/datasets/softmotion_dataset.py

        Only working for bair dataset
        """
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            if self.dataset_name in ['bair', 'googlepush']:
                match = re.search('traj_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            elif self.dataset_name in ['bair_predictions', 'googlepush_predictions']:
                match = re.search('pred_seq_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count
