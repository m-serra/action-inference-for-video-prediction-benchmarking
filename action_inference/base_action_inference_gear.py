import os
import tensorflow as tf
from training_flags import FLAGS
from action_inference.action_inference_model import train_action_inference
import data_readers


class BaseActionInferenceGear(object):

    def __init__(self, model_name, dataset_name, ckpts_dir):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.ckpts_dir = ckpts_dir  # --> set this on the function that runs vp_model?
        self.model_save_dir = None  # --> set this on the function that trains?

        self.context_frames = None
        self.sequence_length = None
        self.n_future = None

    def vp_forward_pass(self, model, input_results, sess):
        """
        inputs
        ------

        outputs
        -------
        - generated frames
        - ground truth actions
        """
        raise NotImplementedError

    def vp_restore_model(self, dataset, mode):
        """
        outputs
        -------
        - model
        - inputs (iterator.get_next() operation)
        - sess (tf.Session())

        """
        raise NotImplementedError

    """SEEMS DONE"""
    def create_predictions_dataset(self, original_dataset, mode, context_frames, sequence_length, predictions_save_dir):

        self.context_frames = context_frames
        self.sequence_length = sequence_length
        self.n_future = sequence_length - context_frames
        predictions_save_dir = os.path.join(predictions_save_dir, original_dataset.dataset_name, self.model_name, mode)

        model, inputs, sess = self.vp_restore_model(dataset=original_dataset, mode=mode)

        num_examples_per_epoch = original_dataset.num_examples_per_epoch(mode=mode)

        PredictionDatasetClass = data_readers.original_to_prediction_map.get(original_dataset.dataset_name)

        sample_ind = 0
        writer = None
        while True:
            if sample_ind >= num_examples_per_epoch:
                break
            try:
                print("saving samples from %d to %d" % (sample_ind, sample_ind + original_dataset.batch_size))

                input_results = sess.run(inputs)
                gt_actions = input_results['action_targets'][:, -(self.n_future-1):, :]
                gt_states = input_results['states'][:, -self.n_future:, :]

                gen_frames = self.vp_forward_pass(model, input_results, sess)

                writer = PredictionDatasetClass.save_tfrecord_example(writer, sample_ind, gen_frames,
                                                                      gt_actions, gt_states, predictions_save_dir)

            except tf.errors.OutOfRangeError:
                break

            sample_ind += original_dataset.batch_size

    def train_inference_model(self, data_reader, n_epochs, seq_len, model_save_dir, shuffle=True,
                              normalize_targets=False, targets_mean=None, targets_std=None):

        d = data_reader

        save_dir = os.path.join(model_save_dir, d.dataset_name, self.model_name)

        # ===== Obtain train data and number of iterations
        train_iterator = d.build_tfrecord_iterator(mode='train')
        train_inputs = train_iterator.get_next()

        if FLAGS.n_iterations is None:
            train_n_iterations = int(d.num_examples_per_epoch(mode='train')/FLAGS.batch_size)
        else:
            train_n_iterations = FLAGS.n_iterations

        # ===== Obtain validation data and number of iterations
        val_iterator = d.build_tfrecord_iterator(mode='val')
        val_inputs = val_iterator.get_next()

        if FLAGS.n_iterations is None:
            val_n_iterations = int(d.num_examples_per_epoch(mode='val') / FLAGS.batch_size)
        else:
            val_n_iterations = FLAGS.n_iterations

        # ===== Preprocess data
        train_image_pairs, train_action_targets = self.preprocess_data(train_inputs,
                                                                       seq_len,
                                                                       shuffle=shuffle,
                                                                       normalize_targets=normalize_targets,
                                                                       targets_mean=targets_mean,
                                                                       targets_std=targets_std)

        val_image_pairs, val_action_targets = self.preprocess_data(val_inputs,
                                                                   seq_len,
                                                                   shuffle=shuffle,
                                                                   normalize_targets=normalize_targets,
                                                                   targets_mean=targets_mean,
                                                                   targets_std=targets_std)

        # sess = tf.Session()
        # sess.run(tf.initialize_all_variables())

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
        raise NotImplementedError

    def evaluate_inference_model(self, datareader):
        """
        """
        raise NotImplementedError

    def evaluate_inference_model_online(self):
        raise NotImplementedError

    def preprocess_data(self, inputs, seq_len, shuffle, normalize_targets, targets_mean=None, targets_std=None):

        """
        Preproccess data for action inference training.
        - Select only
        """

        channels = inputs['images'].get_shape()[-1]
        width = inputs['images'].get_shape()[-2]
        height = inputs['images'].get_shape()[-3]
        paired_seq_len = seq_len - 1  # inputs['images'].get_shape()[-4] # pairing removes one dimension
        batch_size = inputs['images'].get_shape()[-5]

        # ===== Pair every two consecutive images
        images_t = tf.slice(inputs['images'][:, -seq_len:], (0, 0, 0, 0, 0), (-1, paired_seq_len, height, width, channels))
        images_tp1 = tf.slice(inputs['images'][:, -seq_len:], (0, 1, 0, 0, 0), (-1, paired_seq_len, height, width, channels))
        image_pairs = tf.concat([images_t, images_tp1], axis=-1)

        # ===== Normalize data
        # --> decide if this stays because its quite dataset specific
        # --> remove flags
        if normalize_targets:
            assert (targets_mean or targets_std) is not None, \
                'If normalize_data is True data_mean and data_std must be provided'

            # normalize targets by the standard deviation and mean of the respective time step
            std = targets_std[-paired_seq_len:]
            mean = targets_mean[-paired_seq_len:]
            actions_normalized = tf.map_fn(lambda x: x - mean, inputs['action_targets'][-paired_seq_len:])
            actions = tf.map_fn(lambda x: x / std, actions_normalized)
        else:
            actions = inputs['action_targets'][:, -paired_seq_len:]

        # ===== Shuffle data
        if shuffle:
            # indices to apply the same shuffle to inputs and targets
            indices = tf.range(start=0, limit=batch_size * paired_seq_len, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)

            # join batch_dim with seq_dim to mix pairs of all sequences of the batch
            original_image_shape = image_pairs.get_shape()
            original_actions_shape = actions.get_shape()
            actions_stretched = tf.reshape(actions, [-1, 2])  # --> this 2 is action specific
            image_pairs_stretched = tf.reshape(image_pairs, [-1, height, width, 2 * channels])

            # shuffle according to shuffled_indices
            shuffled_image_pairs = tf.gather(image_pairs_stretched, shuffled_indices, axis=0)
            shuffled_actions = tf.gather(actions_stretched, shuffled_indices, axis=0)

            # restore original shape
            image_pairs = tf.reshape(shuffled_image_pairs, original_image_shape)
            actions = tf.reshape(shuffled_actions, original_actions_shape)

        return image_pairs, actions

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
