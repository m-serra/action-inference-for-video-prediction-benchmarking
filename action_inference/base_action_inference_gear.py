import os
import tensorflow as tf
from training_flags import FLAGS
from action_inference.action_inference_model import train_action_inference
import data_readers


class BaseActionInferenceGear(object):

    def __init__(self, model_name, dataset_name, ckpts_dir, sequence_length):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.ckpts_dir = ckpts_dir
        self.model_save_dir = None  # --> set this on the function that trains

        self.context_frames = 2  # --> !!!!!!!!!
        self.sequence_length = sequence_length  # --> !!!!!!!!!
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

    def create_predictions_dataset(self, original_dataset, mode, predictions_save_dir):

        model, inputs, sess = self.vp_restore_model(dataset=original_dataset, mode=mode)

        num_examples_per_epoch = original_dataset.num_examples_per_epoch(mode=mode)

        PredictionDatasetClass = data_readers.original_to_prediction_map.get(original_dataset.dataset_name)

        sample_ind = 0
        while True:
            if sample_ind >= num_examples_per_epoch:
                break
            try:
                print("evaluation samples from %d to %d" % (sample_ind, sample_ind + original_dataset.batch_size))

                input_results = sess.run(inputs)
                gt_actions = input_results['action_targets']

                gen_frames = self.vp_forward_pass(model, input_results, sess)

                PredictionDatasetClass.save_tf_record_example(sample_ind, gen_frames, gt_actions, predictions_save_dir)

            except tf.errors.OutOfRangeError:
                break

        sample_ind += original_dataset.batch_size


    def train_inference_model(self, datareader, n_epochs, normalize_targets=False, targets_mean=None, targets_std=None):

        dataset_dir = os.path.join(self.prediction_dataset_dir, self.model_name)
        save_dir = os.path.join(self.model_save_dir, self.model_name)

        d = datareader

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
        raise NotImplementedError

    def evaluate_inference_model(self, datareader):
        """
        """
        raise NotImplementedError

    def evaluate_inference_model_online(self):
        raise NotImplementedError

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

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))