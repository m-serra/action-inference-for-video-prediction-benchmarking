"""
Code with help from 
 - Alex Lee's https://github.com/alexlee-gk/video_prediction
 - Rachel Finn's https://github.com/m-serra/models/tree/master/video_prediction
"""

import os
import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric
from utils import frechet_video_distance as fvd


class VideoPredictionMetrics(object):

    def __init__(self, model_name, dataset_name, sequence_length, context_frames, save_dir):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.future_length = sequence_length - context_frames
        self.total_psnr_by_step = np.zeros(self.future_length)
        self.total_ssim_by_step = np.zeros(self.future_length)
        self.total_square_psnr_by_step = np.zeros(self.future_length)
        self.total_square_ssim_by_step = np.zeros(self.future_length)
        self.total_fvd = 0
        self.total_square_fvd = 0
        self.n_examples = 0
        self.batch_size = None

        self.fvd_batch_size = 32
        self.all_gt_videos = np.zeros([self.fvd_batch_size, self.future_length, 64, 64, 3])
        self.all_pred_videos = np.zeros([self.fvd_batch_size, self.future_length, 64, 64, 3])
        
        if self.model_name == 'ours_savp':
            model_name = 'savp'
        elif self.model_name == 'ours_vae':
            model_name = 'savp_vae'
        else:
            model_name = self.model_name
        
        self.save_dir = save_dir
        
        # ===== FVD op
        # added this because of an error: https://github.com/tensorflow/tensorflow/issues/6698
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.85

        gen = tf.placeholder(dtype=tf.float32, shape=(self.fvd_batch_size, 28, 64, 64, 3), name='gen')
        true = tf.placeholder(dtype=tf.float32, shape=(self.fvd_batch_size, 28, 64, 64, 3), name='true')

        self.fvd_op = fvd.calculate_fvd(fvd.create_id3_embedding(fvd.preprocess(true, (224, 224))),
                                        fvd.create_id3_embedding(fvd.preprocess(gen, (224, 224))))

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.fvd_list = np.zeros(8)

    def update_metrics_values(self,
                              context_images,
                              gen_images):

        if len(context_images.shape) == 5:
            self.batch_size = context_images.shape[0]
        else:
            self.batch_size = 1

        # only keep the future frames
        gen_images = gen_images[:, -self.future_length:]
        context_images_future = context_images[:, -self.future_length:]
        
        # ===== Compute psnr and ssim for each step of the sequence
        psnr_by_step, ssim_by_step = self.get_seq_psnr_ssim(context_images_future, gen_images)

        # ===== Add to the total psnr/ssim of each timestep
        self.total_psnr_by_step = self.total_psnr_by_step + np.sum(psnr_by_step, axis=0)
        self.total_ssim_by_step = self.total_ssim_by_step + np.sum(ssim_by_step, axis=0)

        # ===== Add to the total square psnr/ssim of each timestep
        self.total_square_psnr_by_step = self.total_square_psnr_by_step + np.sum(np.square(psnr_by_step), axis=0)
        self.total_square_ssim_by_step = self.total_square_ssim_by_step + np.sum(np.square(ssim_by_step), axis=0)
       
        # ==== Register the videos
        self.all_gt_videos[(self.n_examples % self.fvd_batch_size):(self.n_examples % self.fvd_batch_size)+8] = \
                                                                                                context_images_future
        self.all_pred_videos[(self.n_examples % self.fvd_batch_size):(self.n_examples % self.fvd_batch_size)+8] = \
                                                                                                gen_images
        self.n_examples += self.batch_size
        
        # ===== FVD
        if self.n_examples % self.fvd_batch_size == 0:
            fvd_val = self.sess.run(self.fvd_op, {'gen:0': self.all_pred_videos*255, 
                                                  'true:0': self.all_gt_videos*255}) 

            self.fvd_list[int(self.n_examples/self.fvd_batch_size)-1] = fvd_val
            
            self.all_gt_videos = np.zeros([self.fvd_batch_size, self.future_length, 64, 64, 3])
            self.all_pred_videos = np.zeros([self.fvd_batch_size, self.future_length, 64, 64, 3])

    def save_metrics(self):
        
        # ===== PSNR
        avg_psnr_by_step = np.divide(self.total_psnr_by_step, self.n_examples)
        std_psnr_by_step = self.std_dev(self.total_psnr_by_step, self.total_square_psnr_by_step, self.n_examples)
        
        # ===== SSIM
        avg_ssim_by_step = np.divide(self.total_ssim_by_step, self.n_examples)
        std_ssim_by_step = self.std_dev(self.total_ssim_by_step, self.total_square_ssim_by_step, self.n_examples)

        if not os.path.isdir(self.save_dir):
            try:
                original_umask = os.umask(0)
                os.makedirs(self.save_dir, mode=0o777, exist_ok=True)
            finally:
                os.umask(original_umask)

        with open(self.save_dir+'/metrics.pickle', 'wb') as f:
            pickle.dump([avg_psnr_by_step,
                         std_psnr_by_step,
                         avg_ssim_by_step,
                         std_ssim_by_step,
                         np.mean(self.fvd_list)], f)

    def get_seq_psnr_ssim(self, context_frames, gen_frames):

        """Compute PSNR and SSIM from skimage"""

        assert context_frames.shape == gen_frames.shape

        n_channels = gen_frames.shape[-1]
        n_future = gen_frames.shape[1]
        batch_size = gen_frames.shape[0]

        _psnr_by_step = np.zeros([batch_size, n_future])
        _ssim_by_step = np.zeros([batch_size, n_future])

        for b in range(batch_size):
            for t in range(n_future):
                for chan in range(n_channels):
                    _psnr_by_step[b, t] += psnr_metric(context_frames[b, t, :, :, chan], gen_frames[b, t, :, :, chan])
                    _ssim_by_step[b, t] += ssim_metric(context_frames[b, t, :, :, chan], gen_frames[b, t, :, :, chan])

                _psnr_by_step[b, t] /= n_channels
                _ssim_by_step[b, t] /= n_channels

        return _psnr_by_step, _ssim_by_step
    
    def save_inference_metrics(self, y_true, y_pred):
        
        # ===== compute average metrics by sequence
        n_samples = y_true.shape[0]
        r2_by_sequence = np.zeros([n_samples, 2])
        for i in range(n_samples):
            for dim in range(2):
                r2_by_sequence[i, dim] = r2_score(y_true=y_true[i, :, dim],
                                                  y_pred=y_pred[i, :, dim])

        avg_r2 = np.mean(r2_by_sequence)
        avg_l1 = np.mean(abs(y_true - y_pred))

        # ===== compute average metrics by timestep
        avg_r2_by_step_x = np.zeros(self.future_length - 1)
        avg_r2_by_step_y = np.zeros(self.future_length - 1)

        abs_diff_by_step_mean = np.mean(abs(y_true - y_pred), axis=0)
        abs_diff_by_step_std = np.std(abs(y_true - y_pred), axis=0)

        for i in range(self.future_length - 1):
            avg_r2_by_step_x[i] = r2_score(y_true=y_true[:, i, 0],
                                           y_pred=y_pred[:, i, 0])
            avg_r2_by_step_y[i] = r2_score(y_true=y_true[:, i, 1],
                                           y_pred=y_pred[:, i, 1])

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # --> MISSING STANDARD DEVIATIONS
        with open(self.save_dir + '/inference_metrics.pickle', 'wb') as f:
            pickle.dump([abs_diff_by_step_mean,
                         abs_diff_by_step_std,
                         avg_r2_by_step_x,
                         avg_r2_by_step_y,
                         avg_r2,
                         avg_l1], f)

    @staticmethod
    def std_dev(x, x2, n_samples):
        """
        https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
        
        Demonstration: https://clontz.org/blog/2017/04/21/alternate-standard-deviation/
        """
        return [np.sqrt((x2_i / n_samples) - (x_i / n_samples) ** 2) for x_i, x2_i in zip(x, x2)]













