import os
import numpy as np
import tensorflow as tf
from action_inference.base_action_inference_gear import BaseActionInferenceGear


class CDNAGear(BaseActionInferenceGear):

    def __init__(self, **kwargs):
        super(CDNAGear, self).__init__(**kwargs)
        self.itr = 0

    def vp_forward_pass(self, model, inputs, sess, mode=None, feeds=None):

        # a Hack: because I wanted to train CDNA no predict to time step 12 but then wanted to do a second pass over
        # the train set, now predicting to time step 30, with lr=0.0, I defined a parallel model using the same weights
        # and dataset but until time step 30
        if mode == 'train':
            mode = 'get_predictions'

        feed_dict = {mode + '_model/prefix:0': mode,
                     mode + '_model/iter_num:0': np.float32(self.itr),
                     mode + '_model/learning_rate:0': 0.0}

        gen_images, context_images, gt_states = sess.run([feeds['gen'], feeds['context'], feeds['context_st']],
                                                         feed_dict)
        gen_images = gen_images[:, -self.n_future:]
        context_images = context_images[:, -self.n_future:]
        self.itr += 1

        # --> in the future replace states by actions for generality
        return gen_images, context_images, gt_states

    def vp_restore_model(self, dataset, mode, ckpts_dir):

        # a Hack
        if mode == 'train':
            mode = 'get_predictions'

        # ===== Restore pretrained model
        path_to_saved_model = os.path.join(ckpts_dir, 'model')

        sess = tf.Session()
        new_saver = tf.train.import_meta_graph(path_to_saved_model + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(ckpts_dir))
        graph = tf.get_default_graph()

        gen = graph.get_tensor_by_name(mode + '_model/gen_images:0')
        context = graph.get_tensor_by_name(mode + '_model/context_frames:0')
        context_st = graph.get_tensor_by_name(mode + '_model/context_states:0')

        model = graph
        feeds = {'gen': gen, 'context': context, 'context_st': context_st}
        inputs = None

        return model, inputs, sess, feeds

    def evaluate_inference_model_online(self):
        raise NotImplementedError
