import re
import os
import tensorflow as tf
from training_flags import FLAGS
from data_readers.bair_data_reader import BairDataReader


class BairPredictionsDataReader(BairDataReader):

    def __init__(self,
                 model_name,
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
        self.data_dir = os.path.join(self.data_dir, self.dataset_name, model_name)
        self.train_filenames, self.val_filenames, self.test_filenames = self.set_filenames()

    def _parse_sequences(self, serialized_example):
        image_seq, action_target_seq = [], []

        for i in range(self.sequence_length_to_use):

            image_name = str(i) + '/image/encoded'
            # action_target_name = str(i) + '/action_target' # --> change state for actions
            action_target_name = str(i) + '/state_target'


            # --> try to find a better solution for this if like a dummy action target at the end of the dataset
            #if i < self.sequence_length_to_use - 1:
            features = {image_name: tf.FixedLenFeature([1], tf.string),
                        action_target_name: tf.FixedLenFeature([3], tf.float32)}  # -->set target action dim instead of 2
            # else:
            #    features = {image_name: tf.FixedLenFeature([1], tf.string)}

            features = tf.parse_single_example(serialized_example, features=features)

            image = tf.decode_raw(features[image_name], float)
            image = tf.reshape(image, shape=[1, self.ORIGINAL_HEIGHT * self.ORIGINAL_WIDTH * self.COLOR_CHAN])
            image = tf.reshape(image, shape=[self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH, self.COLOR_CHAN])

            assert self.IMG_HEIGHT == self.IMG_WIDTH, 'Unequal height and width unsupported'

            crop_size = min(self.ORIGINAL_HEIGHT, self.ORIGINAL_WIDTH)
            image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
            image = tf.reshape(image, [1, crop_size, crop_size, self.COLOR_CHAN])
            image = tf.image.resize_bicubic(image, [self.IMG_HEIGHT, self.IMG_WIDTH])
            # image = tf.cast(image, tf.float32) / 255.0
            image_seq.append(image)

            # --> try to find a better solution for this if like a dummy action target at the end of the dataset
            # if i < self.sequence_length_to_use - 1:
            state = tf.reshape(features[action_target_name], shape=[1, 3])  # -->set target action dim instead of 2
            action_target_seq.append(state)

        image_seq = tf.concat(image_seq, 0)
        action_target_seq = tf.concat(action_target_seq, 0)

        actions_t = action_target_seq[:-1, :]
        actions_tp1 = action_target_seq[1:, :]
        action_target_seq = actions_tp1[:, :2] - actions_t[:, :2]

        return {'images': image_seq, 'action_targets': action_target_seq}

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

    @staticmethod
    # in the future replace state by actions for generality
    def save_tfrecord_example(writer, example_id, gen_images, gt_state, save_dir):
        """
        SOURCE: https://github.com/OliviaMG/xiaomeng/issues/1

        inputs
        ------
        - gen_images: (batch_size, future_length, h, w, c)
        - gt_actions: (batch_size, future_length-1, dim)
        """

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        batch_size = gen_images.shape[0]
        pred_seq_len = gen_images.shape[1]

        if example_id % 256 == 0:  # save file of 256 examples

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir,
                                    'pred_seq_' + str(example_id) + '_to_' + str(example_id + 255) + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(filename)
            print('Writing', filename)

        # accumulate examples
        feature = {}
        for seq in range(batch_size):
            pred = gen_images[seq]
            st = gt_state[seq]
            # ac = gt_actions[seq]  # --> in the future replace state by actions

            for index in range(pred_seq_len):
                image_raw = pred[index].tostring()
                # encoded_image_string = cv2.imencode('.jpg', traj[index])[1].tostring()
                # image_raw = tf.compat.as_bytes(encoded_image_string)

                feature[str(index) + '/image/encoded'] = _bytes_feature(image_raw)
                feature[str(index) + '/state_target'] = _float_feature(st[index, :].tolist())  # --> replace
                # because actions are considered to be "between frames" there is one less action than frames
                # if index < pred_seq_len - 1:  # --> in the future replace state by actions
                #    feature[str(index) + '/action_target'] = _float_feature(ac[index, :].tolist())

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        return writer
