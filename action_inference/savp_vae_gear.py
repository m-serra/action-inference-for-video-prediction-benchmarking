import os
import json
import tensorflow as tf
from action_inference.base_action_inference_gear import BaseActionInferenceGear
from video_prediction.savp.models import SAVPVideoPredictionModel


class SavpVaeGear(BaseActionInferenceGear):

    def __init__(self, **kwargs):
        super(SavpVaeGear, self).__init__(**kwargs)
        self.num_stochastic_samples = 1

    def vp_forward_pass(self, model, inputs, sess):
        """
        """
        # batch_size = inputs['images'].shape[0]

        # input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
        # --> this is dataset specific
        feed_dict = {'images_ph:0': inputs['images'],
                     'actions_ph:0': inputs['actions'][:, :-1, :],
                     'states_ph:0': inputs['states']}

        for stochastic_sample_ind in range(self.num_stochastic_samples):
            gen_images = sess.run(model.outputs['gen_images'], feed_dict=feed_dict)
            gen_images = gen_images[:, -self.n_future:]

        return gen_images

    def vp_restore_model(self, dataset, mode):
        """
        inputs
        ------
        - mode: controls whether the data that will be fed to the model comes from the 'train' or 'test' set. Doesn't
                control in which mode the model is used, which is always 'test', that is, learning rate = 0
        """
        gpu_mem_frac = 0  # --> check what this does

        iterator = dataset.build_tfrecord_iterator(mode=mode)
        inputs = iterator.get_next()

        checkpoint_dir = os.path.join(self.ckpts_dir, self.dataset_name, 'savp_vae')

        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")

        hparams_dict = dict(model_hparams_dict)

        hparams_dict.update({'context_frames': self.context_frames,  # --> this should be written in a .json in the ckpt dir
                             'sequence_length': self.sequence_length,
                             'repeat': 0})  # --> check what this repeat is

        model = SAVPVideoPredictionModel(mode='test', hparams_dict=hparams_dict)

        # input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
        # --> this is dataset specific
        input_phs = {'images': tf.placeholder(inputs['images'].dtype, inputs['images'].shape, 'images_ph'),
                     'actions': tf.placeholder(inputs['actions'].dtype,
                                               [dataset.batch_size, self.sequence_length-1, 4], 'actions_ph'),
                     'states': tf.placeholder(inputs['states'].dtype, inputs['states'].shape, 'states_ph')}

        with tf.variable_scope(''):
            model.build_graph(input_phs)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        sess = tf.Session(config=config)
        sess.graph.as_default()

        model.restore(sess, checkpoint_dir)

        return model, inputs, sess

    def train_inference_model_online(self):
        raise NotImplementedError

    def evaluate_inference_model(self, datareader):
        """
        """
        raise NotImplementedError

    def evaluate_inference_model_online(self):
        raise NotImplementedError