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
        batch_size = inputs['images'].shape[0]

        # input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}
        # --> this is dataset specific
        input_phs = {'images': tf.placeholder(inputs['images'].dtype, inputs['images'].shape, 'images_ph'),
                     'actions': tf.placeholder(inputs['actions'].dtype,
                                               [batch_size, self.sequence_length-1, 4], 'actions_ph'),
                     'states': tf.placeholder(inputs['states'].dtype, inputs['states'].shape, 'states_ph')}

        feed_dict = {input_ph: inputs[name].as_type(float) for name, input_ph in input_phs.items()}
        print('IMAGES:')
        print(feed_dict[input_phs['images']].shape)
        # a little hack because my iterator returns the last time step's action
        feed_dict[input_phs['actions']] = inputs['actions'][:, :-1, :]
        print(feed_dict.keys())
        for stochastic_sample_ind in range(self.num_stochastic_samples):
            gen_images = sess.run(model.outputs['gen_images'], feed_dict=feed_dict)

        return gen_images

    def vp_restore_model(self, dataset, mode):

        gpu_mem_frac = 0  # --> check what this does

        iterator = dataset.build_tfrecord_iterator(mode='test')
        inputs = iterator.get_next()

        checkpoint_dir = os.path.join(self.ckpts_dir, self.dataset_name, 'savp_vae')

        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
            print(checkpoint_dir)

        #self.context_frames = dataset_hparams_dict['context_frames']
        #self.sequence_length = dataset_hparams_dict['sequence_length']
        #self.n_future = self.sequence_length - self.context_frames

        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")

        hparams_dict = dict(model_hparams_dict)

        hparams_dict.update({'context_frames': self.context_frames,  # --> this should be written in a .json in the ckpt dir
                             'sequence_length': self.sequence_length,
                             'repeat': 0})  # --> check what this repeat is

        model = SAVPVideoPredictionModel(mode=mode, hparams_dict=hparams_dict)

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