import re
import os
import tensorflow as tf
from training_flags import FLAGS
from data_readers.bair_data_reader import BairDataReader

class BairPredictionsDataReader(BairDataReader):

    def __init__(self,
                 dataset_dir=None,
                 *args,
                 **kwargs):
        """
        Dataset class for the BAIR and Google Push datasets.
        :param dataset_dir: (str, optional) path to dataset directory, containing the /train and a /test directories.
                            Defaults to FLAGS.bair_dir or FLAGS.google_dir, defined in training_flags.py, depending on
                            the dataset_name parameter.
        """
        super(BairPredictionsDataReader, self).__init__(*args, **kwargs)
        self.dataset_name = 'bair_predictions'
        self.data_dir = dataset_dir if dataset_dir else FLAGS.bair_predictions_dir
        self.train_filenames, self.val_filenames, self.test_filenames = self.set_filenames()

    def _parse_prediction_sequences(self, serialized_example):
        image_seq, state_seq = [], []
        for i in range(self.sequence_length_to_use):

            image_name = str(i) + '/image_aux1/encoded'
            state_name = str(i) + '/endeffector_pos'

            features = {image_name: tf.FixedLenFeature([1], tf.string),
                        state_name: tf.FixedLenFeature([self.STATE_DIM], tf.float32)}
            features = tf.parse_single_example(serialized_example, features=features)

            image = tf.decode_raw(features[image_name], float)
            image = tf.reshape(image, shape=[1, self.IMG_HEIGHT * self.IMG_WIDTH * self.COLOR_CHAN])
            image = tf.reshape(image, shape=[self.IMG_HEIGHT * self.IMG_WIDTH * self.COLOR_CHAN])
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

        return {'images': image_seq, 'action_targets': delta_xy}

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
            match = re.search('pred_seq_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count
