import os
import tensorflow as tf
from training_flags import FLAGS
from data_readers.bair_data_reader import DataReader
from action_inference_model import train_action_inference


class BaseActionInference(object):

    def __init__(self, model_name, dataset_name, prediction_dataset_dir, model_save_dir):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.prediction_dataset_dir = prediction_dataset_dir
        self.model_save_dir = model_save_dir
        self.seq_len = FLAGS.train_sequence_length
        self.test_seq_len = FLAGS.test_sequence_length

    def vp_forward_pass(self):
        """
        inputs
        ------

        outputs
        -------
        - generated frames
        - ground truth actions
        """
        pass

    def vp_restore_model(self):
        pass

    def train_inference_model(self, n_epochs, normalize_targets=False, targets_mean=None, targets_std=None):

        dataset_dir = os.path.join(self.prediction_dataset_dir, self.model_name)
        save_dir = os.path.join(self.model_save_dir, self.model_name)

        d = DataReader(dataset_name=self.dataset_name,
                       dataset_dir=dataset_dir,
                       shuffle=True,
                       sequence_length_train=self.seq_len,
                       sequence_length_test=self.test_seq_len)

        # ===== Obtain train data and number of iterations
        train_iterator = d.build_tfrecord_iterator(mode='train')
        train_inputs = train_iterator.get_next()

        if FLAGS.n_iterations is None:
            train_n_iterations = int(d.num_examples_per_epoch()/FLAGS.batch_size)
        else:
            train_n_iterations = FLAGS.n_iterations

        # ===== Obtain validation data and number of iterations
        val_iterator = d.build_tfrecord_iterator(mode='val')
        val_inputs = val_iterator.get_next()

        if FLAGS.n_iterations is None:
            val_n_iterations = int(d.num_examples_per_epoch() / FLAGS.batch_size)
        else:
            val_n_iterations = FLAGS.n_iterations

        # ===== Preprocess data
        train_image_pairs, train_action_targets = self.preprocess_data(train_inputs,
                                                                       shuffle=True,
                                                                       normalize_targets=normalize_targets,
                                                                       targets_mean=targets_mean,
                                                                       targets_std=targets_std)

        val_image_pairs, val_action_targets = self.preprocess_data(val_inputs,
                                                                   shuffle=True,
                                                                   normalize_targets=normalize_targets,
                                                                   targets_mean=targets_mean,
                                                                   targets_std=targets_std)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        # --> allow choice of callbacks
        # ===== Train action inference model
        model, history = train_action_inference(train_image_pairs,
                                                train_action_targets,
                                                epochs=n_epochs,
                                                steps_per_epoch=train_n_iterations,
                                                val_inputs=val_image_pairs,
                                                val_targets=val_action_targets,
                                                validation_steps=val_n_iterations,
                                                save_path=save_dir)

    def train_inference_model_online(self):
        pass

    def evaluate_inference_model(self):
        """
        """
        # ===== instance a metrics object
        metrics = VideoPredictionMetrics(model_name=self.model_name,
                                         dataset_name=self.dataset_name,
                                         sequence_length=30,  # --> be careful because I subtract the context frames inside
                                         context_frames=FLAGS.context_frames)  # --> avoid having FLAGS in this file

        # ===== create a dataset object for the bair predictions of the current model
        model_dataset_dir = os.path.join(self.dataset_dir, self.model_name)

        # --> split data readers into subcalsses and make a suclass of bair predictoins
        d = DataReader(dataset_name='bair_predictions',
                       dataset_dir=model_dataset_dir,
                       dataset_repeat=1,
                       shuffle=False,
                       sequence_length_train=12,
                       sequence_length_test=sequence_length)

    def evaluate_inference_model_online(self):
        pass

    @staticmethod
    def preprocess_data(inputs, shuffle, normalize_targets, targets_mean=None, targets_std=None):

        channels = inputs['images'].get_shape[-1]
        width = inputs['images'].get_shape[-2]
        height = inputs['images'].get_shape[-3]
        seq_len = inputs['images'].get_shape[-4] - 1  # pairing will remove one dimension
        batch_size = inputs['images'].get_shape[-5]

        # ===== Pair every two consecutive images
        images_t = tf.slice(inputs['images'], (0, 0, 0, 0, 0), (-1, seq_len, height, width, channels))
        images_tp1 = tf.slice(inputs['images'], (0, 1, 0, 0, 0), (-1, seq_len, height, width, channels))
        image_pairs = tf.concat([images_t, images_tp1], axis=-1)

        # ===== Normalize data
        # --> decide if this stays because its quite dataset specific
        if normalize_targets:
            assert (targets_mean or targets_std) is not None, \
                'If normalize_data is True data_mean and data_std must be provided'

            # normalize targets by the standard deviation and mean of the respective time step
            std = targets_std[FLAGS.context_frames:FLAGS.context_frames + seq_len]
            mean = targets_mean[FLAGS.context_frames:FLAGS.context_frames + seq_len]
            actions_normalized = tf.map_fn(lambda x: x - mean, inputs['action_targets'])
            actions = tf.map_fn(lambda x: x / std, actions_normalized)
        else:
            # --> change the name of from deltas to action_targets
            actions = inputs['action_targets']

        # ===== Shuffle data
        if shuffle:
            # indices to apply the same shuffle to inputs and targets
            indices = tf.range(start=0, limit=batch_size * seq_len, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)

            # join batch_dim with seq_dim to mix pairs of all sequences of the batch
            original_image_shape = image_pairs.get_shape()
            original_actions_shape = actions.get_shape()
            actions_stretched = tf.reshape(actions, [-1, 2])
            image_pairs_stretched = tf.reshape(image_pairs, [-1, height, width, 2 * channels])

            # shuffle according to shuffled_indices
            shuffled_image_pairs = tf.gather(image_pairs_stretched, shuffled_indices, axis=0)
            shuffled_actions = tf.gather(actions_stretched, shuffled_indices, axis=0)

            # restore original shape
            image_pairs = tf.reshape(shuffled_image_pairs, original_image_shape)
            actions = tf.reshape(shuffled_actions, original_actions_shape)

        return image_pairs, actions
