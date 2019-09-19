import os
import sys
import multiprocessing
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


class BaseDataReader(object):

    def __init__(self,
                 batch_size,
                 sequence_length_train,
                 sequence_length_test,
                 shuffle=False,
                 dataset_repeat=None):
        """
        Dataset class for the BAIR and Google Push datasets.

        inputs
        ------
        - batch_size: (int) size of the mini batches the iterator will provide
        - sequence_length_train: (int) number of timesteps to use for training and validation
        - sequence_length_test: (int) number of timesteps to use for test
        - shuffle: (boolean) whether to shuffle the train/val filenames and tfrecord samples. Defaults
                        to FLAGS.shuffle.
        - dataset_repeat: (int) number of times the dataset can be iterated. If None indefinite iteration is
                                allowed.
        """

        self.COLOR_CHAN = None
        self.IMG_WIDTH = None
        self.IMG_HEIGHT = None
        self.STATE_DIM = None
        self.ACTION_DIM = None
        self.ORIGINAL_WIDTH = None
        self.ORIGINAL_HEIGHT = None
        self.filenames = None
        self.data_dir = None
        self.train_val_split = None
        self.train_filenames = None
        self.val_filenames = None
        self.test_filenames = None
        self.shuffle = shuffle
        self.n_threads = multiprocessing.cpu_count()

        self.dataset_repeat = dataset_repeat
        self.batch_size = batch_size
        self.sequence_length_train = sequence_length_train
        self.sequence_length_test = sequence_length_test
        self.sequence_length_to_use = None

    def set_filenames(self):

        train_val_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        try:
            train_val_filenames = gfile.Glob(os.path.join(train_val_dir, '*'))
            idx_train_val_split = int(np.floor(self.train_val_split * len(train_val_filenames)))
            train_filenames = train_val_filenames[:idx_train_val_split]
            val_filenames = train_val_filenames[idx_train_val_split:]
        except:
            train_filenames = None
            val_filenames = None
            print("No train/val data files found.")

        try:
            test_filenames = gfile.Glob(os.path.join(test_dir, '*'))
        except:
            test_filenames = None
            print("No test data files found.")

        return train_filenames, val_filenames, test_filenames

    def build_tfrecord_iterator(self, mode='train'):
        """Create input tfrecord iterator
        Args:
        -----
            mode: 'train', 'val' or 'test'

        Returns:
        --------
            An iterator that return images, actions, states, distances, angles
        """
        assert mode in ['train', 'val', 'test'], 'Mode must be one of "train", "val" or "test"'

        if mode == 'train' and self.train_filenames is not None:
            self.filenames = self.train_filenames
            self.sequence_length_to_use = self.sequence_length_train
        elif mode == 'val' and self.val_filenames is not None:
            self.filenames = self.val_filenames
            self.sequence_length_to_use = self.sequence_length_train
        elif self.test_filenames:
            self.filenames = self.test_filenames
            self.sequence_length_to_use = self.sequence_length_test

        if self.filenames is None:
            print('No files found')
            sys.exit(0)

        filename_queue = tf.data.Dataset.from_tensor_slices(self.filenames)
        if self.shuffle and mode != 'test':
            filename_queue = filename_queue.shuffle(buffer_size=len(self.filenames))

        dataset = tf.data.TFRecordDataset(filename_queue)

        if self.shuffle and mode is not 'test':
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.dataset_repeat))
        else:
            dataset = dataset.repeat(count=self.dataset_repeat)  # to allow multiple epochs with one_shot_iterator

        dataset = dataset.map(lambda x: self._parse_sequences(x))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        iterator = dataset.make_one_shot_iterator()

        return iterator

    def _parse_sequences(self, serialized_example):
        pass

    def num_examples_per_epoch(self, mode):
        pass
