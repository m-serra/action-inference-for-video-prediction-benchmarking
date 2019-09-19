# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for building the input for the prediction model.

"""

import os
import tensorflow as tf
from training_flags import FLAGS
from .base_data_reader import BaseDataReader


class GooglePushDataReader(BaseDataReader):

    def __init__(self,
                 dataset_dir=None,
                 *args,
                 **kwargs):
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
        super(GooglePushDataReader, self).__init__(*args, **kwargs)
        self.COLOR_CHAN = 3
        self.IMG_WIDTH = 64
        self.IMG_HEIGHT = 64
        self.STATE_DIM = 5
        self.ACTION_DIM = 5
        self.ORIGINAL_WIDTH = 640
        self.ORIGINAL_HEIGHT = 512
        self.data_dir = dataset_dir if dataset_dir else FLAGS.google_dir
        self.train_filenames, self.val_filenames, self.test_filenames = self.set_filenames()

    def _parse_sequences(self, serialized_example):

        image_seq, state_seq, action_seq = [], [], []

        for i in range(self.sequence_length_to_use):

            image_name = 'move/' + str(i) + '/image/encoded'
            action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
            state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'

            features = {image_name: tf.FixedLenFeature([1], tf.string),
                        action_name: tf.FixedLenFeature([self.ACTION_DIM], tf.float32),
                        state_name: tf.FixedLenFeature([self.STATE_DIM], tf.float32)}

            features = tf.parse_single_example(serialized_example, features=features)

            image_buffer = tf.reshape(features[image_name], shape=[])
            image = tf.image.decode_jpeg(image_buffer, channels=self.COLOR_CHAN)
            image.set_shape([self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH, self.COLOR_CHAN])

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
            return {'images': image_seq,
                    'actions': action_seq,
                    'states': state_seq}
        else:
            zeros_action = tf.zeros([self.sequence_length_to_use, self.ACTION_DIM])
            zeros_state = tf.zeros([self.sequence_length_to_use, self.STATE_DIM])
            return {'images': image_seq,
                    'actions': zeros_action,
                    'states': zeros_state}

    def num_examples_per_epoch(self, mode):
        """
        SOURCE:
        https://github.com/alexlee-gk/video_prediction/blob/master/video_prediction/datasets/google_robot_dataset.py
        """
        # --> NEED TO TEST THIS
        if os.path.basename(self.input_dir) == 'push_train':
            count = 51615
        elif os.path.basename(self.input_dir) == 'push_testseen':
            count = 1038
        elif os.path.basename(self.input_dir) == 'push_testnovel':
            count = 995
        else:
            raise NotImplementedError
        return count


