import os
import numpy as np
import tensorflow as tf
from training_flags import FLAGS
from action_inference.action_inference_model import train_action_inference
import data_readers

from action_inference.action_inference_model import action_inference_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from metrics import VideoPredictionMetrics


class BaseActionInferenceGear(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.ckpts_dir = None
        self.model_save_dir = None

        self.context_frames = None
        self.sequence_length = None
        self.n_future = None

    def vp_forward_pass(self, model, input_results, sess, mode=None, feeds=None):
        """
        To be implemented by subclasses.

        inputs
        ------

        outputs
        -------
        - generated frames
        - ground truth actions
        """
        raise NotImplementedError

    def vp_restore_model(self, dataset, mode, ckpts_dir):
        """
        To be implemented by subclasses.

        Inputs
        ------
        - dataset (object): an object of a subclass of BaseDataReader
        - mode (str): either 'train' or 'test'
        - ckpts_dir (str): path to a directory containing a 'original_dataset_name/model_name/' subdirectory with the
                           necessary files to load the pretrained video prediction model, e.g. a .meta file

        Outputs
        -------
        - model
        - inputs
        - sess

        """
        raise NotImplementedError

    def create_predictions_dataset(self, original_dataset, mode, context_frames, predictions_save_dir,
                                   vp_ckpts_dir, num_examples=None):
        """
        This function has 3 essential steps:
        1: run vp_restore_model(): loads a pretrained video prediction model from
                                   'vp_ckpts_dir/original_dataset_name/model_name'. This method is defined in subclasses
                                   of BaseActionInferenceGear.
        2: run vp_forward_pass(): iteratively performs forward passes on the loaded video prediction model, giving as
                                  input frames (+ actions + states) from the original train/test dataset provided.
                                  The number of context frames is an input argument. This method is defined in
                                  subclasses of BaseActionInferenceGear.
        3: run save_tfrecord_example(): saves the video predictions made to .tfrecord files, along with other features
                                        such as actions or states. This method is defined in subclasses of
                                        BaseDataReader.

        inputs
        ------
        - original_dataset (object): an object of a subclass of BaseDataReader.
        - mode (str): either 'train' or 'test'
        - context_frames (int): the number of context frames the video prediction model will observe for each prediction
        - predictions_save_dir (str): directory where the tfrecord files with video predictions will be saved
        - vp_ckpts_dir (str): path to a directory containing a 'original_dataset_name/model_name/' subdirectory with the
                              necessary files to load the pretrained video prediction model, e.g. a .meta file
        - num_examples (int): number of examples to save. If None the whole original_dataset will be used.
        """

        assert mode in ['train', 'test'], "Mode must be either 'train' or 'test'"

        self.context_frames = context_frames

        if mode == 'test':
            self.sequence_length = original_dataset.sequence_length_test
        else:
            self.sequence_length = original_dataset.sequence_length_train

        self.n_future = self.sequence_length - self.context_frames

        ckpts_dir = os.path.join(vp_ckpts_dir, original_dataset.dataset_name, self.model_name)
        predictions_save_dir = os.path.join(predictions_save_dir, original_dataset.dataset_name + '_predictions',
                                            self.model_name, mode)

        model, inputs, sess, feeds = self.vp_restore_model(dataset=original_dataset, mode=mode, ckpts_dir=ckpts_dir)

        if num_examples is None:
            num_examples = original_dataset.num_examples_per_epoch(mode=mode)

        PredictionDatasetClass = data_readers.original_to_prediction_map.get(original_dataset.dataset_name)

        sample_ind = 0
        writer = None
        while True:
            if sample_ind >= num_examples:
                break
            try:
                print("saving samples from %d to %d" % (sample_ind, sample_ind + original_dataset.batch_size))
                # in the future replace states by actions for generality
                # gen_frames, gt_actions = self.vp_forward_pass(model, inputs, sess, mode, feeds)
                gen_frames, _, gt_states = self.vp_forward_pass(model, inputs, sess, mode, feeds)

                writer = PredictionDatasetClass.save_tfrecord_example(writer, sample_ind, gen_frames,
                                                                      gt_states, predictions_save_dir)

            except tf.errors.OutOfRangeError:
                break

            sample_ind += original_dataset.batch_size

    def train_inference_model(self, data_reader, model_save_dir, n_epochs=10,
                              train_num_examples=None, val_num_examples=None):
        """
        This function has 3 essential steps:
        1: run data_reader.build_tfrecord_iterator() and iterator.get_next(): The first method creates a Tensorflow
                        Dataset Iterator, which reads parses .tfrecord files and performs some data preprocessing. The
                        second method declares the operation that fetches the next sample of the dataset, using the
                        iterator. This operation is passed to Keras.train(), which offers the option of training with
                        iterators/generators. This is repeated for train and validation.
        2: run preprocess_data(): this method performs some data prepossessing necessary for training the inference
                        model, namely, pair every two consecutive frames in the shape (bs, seq_len-1, h, w, 6). Again,
                        this is done both for the training and validation data.
        3: run train_action_inference_model(): this method trains the action inference model, as defined in
                                               action_inference_model.py. By default the model parameters that gave
                                               the best validation loss are selected to be saved to the model checkpoint


        Inputs
        ------
        - data_reader (object): an object of a subclass of BaseDataReader
        - model_save_dir (str): directory where the trained model checkpoint will be saved
        - n_epochs (int): number of times the dataset will be passed for training.
        - train_num_examples (int): number of examples to train on. If None the whole dataset will be used.
        - val_num_examples (int): number of examples to validate on. If None the whole dataset will be used.
        """

        self.sequence_length = data_reader.sequence_length_train
        save_dir = os.path.join(model_save_dir, data_reader.dataset_name, self.model_name)

        # ===== Obtain train data and number of iterations
        train_iterator = data_reader.build_tfrecord_iterator(mode='train')
        train_inputs = train_iterator.get_next()

        if train_num_examples is None:
            train_num_examples = int(data_reader.num_examples_per_epoch(mode='train')/data_reader.batch_size)

        # ===== Obtain validation data and number of iterations
        val_iterator = data_reader.build_tfrecord_iterator(mode='val')
        val_inputs = val_iterator.get_next()

        if val_num_examples is None:
            val_num_examples = int(data_reader.num_examples_per_epoch(mode='val') / data_reader.batch_size)

        # ===== Preprocess data
        train_image_pairs, train_action_targets = self._preprocess_data(train_inputs, shuffle=True)
        val_image_pairs, val_action_targets = self._preprocess_data(val_inputs, shuffle=False)

        # --> allow choice of callbacks
        # ===== Train action inference model
        model, history = train_action_inference(train_image_pairs,
                                                train_action_targets,
                                                epochs=n_epochs,
                                                steps_per_epoch=train_num_examples,
                                                val_inputs=val_image_pairs,
                                                val_targets=val_action_targets,
                                                validation_steps=val_num_examples,
                                                save_path=save_dir)

    """Not finished"""
    def train_inference_model_online(self, dataset, context_frames, sequence_length, n_epochs, vp_ckpts_dir,
                                     model_save_dir, shuffle=True):
        self.context_frames = context_frames
        self.sequence_length = sequence_length
        self.n_future = sequence_length - context_frames
        ckpts_dir = os.path.join(vp_ckpts_dir, dataset.dataset_name, self.model_name)
        model_save_dir = os.path.join(vp_ckpts_dir, dataset.dataset_name, 'savp_online')
        model, inputs, sess = self.vp_restore_model(dataset=dataset, mode='train', ckpts_dir=ckpts_dir)

        num_examples_per_epoch = dataset.num_examples_per_epoch(mode='train')

        pred_images = model.outputs['gen_images']
        gt_actions = inputs['action_targets'][:, -(self.n_future - 1):, :]

        # ===== Preprocess data
        train_image_pairs, train_action_targets = self.preprocess_data(inputs={'images': pred_images,
                                                                               'action_targets': gt_actions},
                                                                       seq_len=self.n_future,
                                                                       shuffle=shuffle,
                                                                       normalize_targets=False)

        # ===== Train action inference model
        # --> add validation
        model, history = train_action_inference(train_image_pairs,
                                                train_action_targets,
                                                epochs=n_epochs,
                                                steps_per_epoch=num_examples_per_epoch,
                                                # val_inputs=val_image_pairs,
                                                # val_targets=val_action_targets,
                                                # validation_steps=val_n_iterations,
                                                save_path=model_save_dir)

    def evaluate_inference_model(self, data_reader, model_ckpt_dir, results_save_dir, num_examples=None):
        """
        Does forward pass on the test set of data_reader using the pre-trained action inference model in model_ckpt_dir.
        Results are saved to pickle files in results_save_dir. The list of saved results is
        [MAE_by_step_mean, MAE_by_step_std, avg_r2_by_step_x, avg_r2_by_step_y, avg_r2, avg_MAE].
        The sequence length used for evaluation is data_reader.sequence_length.

        Inputs
        ------
        - data_reader (object): an object of a subclass of BaseDataReader
        - model_ckpt_dir (str): path to a directory containing a 'dataset_name/model_name/' subdirectory with the
                               necessary files to load the pretrained inference model, e.g., a .h5 file
        - results_save_dir (str): directory where the pickle files with metric results will be saved.
        - num_examples (int): number of examples to evaluate on. If None the whole dataset will be used.
        """

        assert data_reader.batch_size == 1, 'During evaluation batch size should be 1.'

        self.sequence_length = data_reader.sequence_length_test
        results_save_dir = os.path.join(results_save_dir, data_reader.dataset_name, self.model_name)
        model_ckpt_dir = os.path.join(model_ckpt_dir, data_reader.dataset_name, self.model_name)

        # ===== Obtain train data and number of iterations
        test_iterator = data_reader.build_tfrecord_iterator(mode='test')
        test_inputs = test_iterator.get_next()

        if num_examples is None:
            num_examples = int(data_reader.num_examples_per_epoch(mode='test'))

        # ===== instance metrics class
        metrics = VideoPredictionMetrics(model_name=self.model_name,
                                         dataset_name=data_reader.dataset_name,
                                         sequence_length=self.sequence_length,
                                         context_frames=0,  # --> !!!!!!!!!!!!
                                         save_dir=results_save_dir)

        # ===== preprocess data
        test_image_pairs, test_action_targets = self._preprocess_data(test_inputs, shuffle=False)
        action_size = test_action_targets.get_shape()[-1]
        paired_sequence_length = test_action_targets.get_shape()[-2]

        # ===== restore the trained inference model
        model = self._restore_inference_model(test_image_pairs, model_ckpt_dir)

        # ===== forward pass on test set
        all_pred_seq = np.zeros([num_examples, paired_sequence_length, action_size])
        all_gt_seq = np.zeros([num_examples, paired_sequence_length, action_size])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(num_examples):
            try:
                imgs_gt, gt_targets = sess.run([test_image_pairs, test_action_targets])
                inferred_actions = model.predict(imgs_gt, steps=1)

                all_pred_seq[i] = inferred_actions[-self.sequence_length:]
                all_gt_seq[i] = gt_targets[-self.sequence_length:]

            except tf.errors.OutOfRangeError:
                print('end of dataset. %d iterations' % i)
                break

        metrics.save_inference_metrics(all_gt_seq, all_pred_seq)

    def evaluate_inference_model_online(self):
        raise NotImplementedError

    def evaluate_vp_model(self, data_reader, context_frames, vp_ckpts_dir, results_dir, num_examples=None):
        """
        This function has 3 essential steps:
        1: run vp_restore_model(): loads a pretrained video prediction model from
                                   'vp_ckpts_dir/original_dataset_name/model_name'. This method is defined in subclasses
                                   of BaseActionInferenceGear.
        2: run vp_forward_pass(): iteratively performs forward passes on the loaded video prediction model, giving as
                                  input frames (+ actions + states) from the original train/test dataset provided.
                                  The number of context frames is an input argument. This method is defined in
                                  subclasses of BaseActionInferenceGear.
        3: run update_metrics_values() and save_metrics(): iteratively computes metrics and saves them to a pickle file
                                  at the end. The saved variables list is:
                                  [avg_psnr_by_step, std_psnr_by_step, avg_ssim_by_step, std_ssim_by_step, fvd)

        Inputs
        ------
        - data_reader (object): an object of a subclass of BaseDataReader
        - context_frames (int): the number of context frames the video prediction model will observe for each prediction
        - vp_ckpts_dir (str): path to a directory containing a 'dataset_name/model_name/' subdirectory with the
                              necessary files to load the pretrained video prediction model, e.g. a .meta file
        - results_dir (str): directory where the pickle files with metric results will be saved.
        - num_examples (int): number of examples to save. If None the whole original_dataset will be used.
        """

        self.context_frames = context_frames
        self.sequence_length = data_reader.sequence_length_test
        self.n_future = self.sequence_length - self.context_frames
        ckpts_dir = os.path.join(vp_ckpts_dir, data_reader.dataset_name, self.model_name)
        results_dir = os.path.join(results_dir, data_reader.dataset_name, self.model_name)

        model, inputs, sess, feeds = self.vp_restore_model(dataset=data_reader, mode='test', ckpts_dir=ckpts_dir)

        if num_examples is None:
            num_examples = data_reader.num_examples_per_epoch(mode='test')

        metrics = VideoPredictionMetrics(model_name=self.model_name,
                                         dataset_name=data_reader.dataset_name,
                                         sequence_length=self.sequence_length,
                                         context_frames=self.context_frames,
                                         save_dir=results_dir)

        sample_ind = 0
        while True:
            if sample_ind >= num_examples:
                break
            try:
                print("evaluating samples from %d to %d" % (sample_ind, sample_ind + data_reader.batch_size))

                gen_frames, gt_frames, _ = self.vp_forward_pass(model, inputs, sess, feeds=feeds, mode='test')

                metrics.update_metrics_values(context_images=gt_frames,
                                              gen_images=gen_frames)

            except tf.errors.OutOfRangeError:
                break

            sample_ind += data_reader.batch_size

        metrics.save_metrics()

    def _preprocess_data(self, inputs, shuffle):
        """
        Prepossesses data for action inference training. In particular, every two consecutive input images are paired,
        e.g., (im_0, im_1), (im_1, im_2), (im_2, im_3), ... The pairing is made by concatenation in the channels
        dimension. So for input['images'] of shape (batch_size, sequence_length, height, width, channels) the output
        image pairs will be of shape (batch_size, sequence_length-1, height, width, 2*channels).
        The same applies for the actions targets.

        Inputs
        ------
        - inputs (dict): Dictionary with 'images' and 'action_targets'. Images should be 5D - (batch_size,
                         sequence_length, height, width, channels) -  and action_targets should be 3D - (batch_size,
                         sequence_length, action_size)
        - shuffle (bool): If true image pairs and targets are shuffled. To shuffle, the dimensions batch_size and
                          sequence_length of the inputs are first collapsed together into one dimension of size:
                          batch_size*sequence_length. The inputs and targets are then shuffled along this new dimension.
                          Finally the original shape is recovered, now with pairs originating from different batches and
                          different  time steps.

        Outputs
        -------
        - image_pairs (tensor): Paired image pairs of shape (batch_size, sequence_length-1, height, width, 2*channels)
        - action_pairs (tensor): Paired action pairs of shape (batch_size, sequence_length-1, action_size)
        """

        channels = inputs['images'].get_shape()[-1]
        width = inputs['images'].get_shape()[-2]
        height = inputs['images'].get_shape()[-3]
        sequence_length = inputs['images'].get_shape()[-4]
        paired_seq_len = sequence_length - 1  # pairing removes one dimension
        batch_size = inputs['images'].get_shape()[-5]

        # ===== Pair every two consecutive images
        images_t = tf.slice(inputs['images'][:, -sequence_length:], (0, 0, 0, 0, 0), (-1, paired_seq_len, height, width, channels))
        images_tp1 = tf.slice(inputs['images'][:, -sequence_length:], (0, 1, 0, 0, 0), (-1, paired_seq_len, height, width, channels))
        image_pairs = tf.concat([images_t, images_tp1], axis=-1)

        action_pairs = inputs['action_targets'][:, -paired_seq_len:]
        action_size = inputs['action_targets'].get_shape()[-1]

        # ===== Shuffle data
        if shuffle:
            # indices to apply the same shuffle to inputs and targets
            indices = tf.range(start=0, limit=batch_size * paired_seq_len, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)

            # join batch_dim with seq_dim to mix pairs of all sequences of the batch
            original_image_shape = image_pairs.get_shape()
            original_actions_shape = action_pairs.get_shape()
            actions_stretched = tf.reshape(action_pairs, [-1, action_size])
            image_pairs_stretched = tf.reshape(image_pairs, [-1, height, width, 2 * channels])

            # shuffle according to shuffled_indices
            shuffled_image_pairs = tf.gather(image_pairs_stretched, shuffled_indices, axis=0)
            shuffled_actions = tf.gather(actions_stretched, shuffled_indices, axis=0)

            # restore original shape
            image_pairs = tf.reshape(shuffled_image_pairs, original_image_shape)
            action_pairs = tf.reshape(shuffled_actions, original_actions_shape)

        return image_pairs, action_pairs

    def _restore_inference_model(self, input_image_pairs, ckpt_dir):

        """
        Restores an pre-trained inference model from a given checkpoint.
        """

        c = input_image_pairs.get_shape()[-1]
        w = input_image_pairs.get_shape()[-2]
        h = input_image_pairs.get_shape()[-3]
        seq_len = input_image_pairs.get_shape()[-4]

        weight_path = os.path.join(ckpt_dir, 'model_weights.h5')

        model_input = Input(shape=(seq_len,  h, w, c))
        inferred_actions = action_inference_model(model_input)

        model = Model(inputs=model_input, outputs=inferred_actions)
        model.load_weights(weight_path)

        return model
