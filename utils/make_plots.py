import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from training_flags import FLAGS

import tensorflow as tf
from data_readers.bair_predictions_data_reader import BairPredictionsDataReader
from action_inference.action_inference_model import action_inference_model
from training_flags import FLAGS
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import backend as K

from data_readers.bair_data_reader import BairDataReader
from sklearn.metrics import r2_score

SMALL_SIZE = 20
MEDIUM_SIZE =24
BIGGER_SIZE = 40
LINEWIDTH = 4.0

color_mean = ['orangered', 'mediumvioletred', 'teal', 'gold', 'mediumblue']
color_shading = ['lightpink', 'thistle', 'paleturquoise', 'lemonchiffon', 'azure']


def plot_mean_and_CI(mean, standard_deviation, n_samples, color_mean=None, color_shading=None, labels=None,
                     start=2, end=30, step=1, linestyle='solid', linewidth=4.0):
    z = 1.96  # for 95% confidence interval

    ub = z * (standard_deviation / np.sqrt(n_samples))  # + mean
    lb = z * (standard_deviation / np.sqrt(n_samples))  # + mean

    # plt.fill_between(range(start, end, step), ub, lb, color=color_shading, alpha=.5)
    # plt.plot(range(start, end, step), mean, color_mean, label=labels, linestyle=linestyle, linewidth=linewidth)
    plt.errorbar(x=range(start, end, step), 
                 y=mean, 
                 yerr=[lb, ub], 
                 color=color_mean, 
                 label=labels, 
                 linestyle=linestyle,
                 linewidth=linewidth)


def configure_plt():
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def make_plots_action_inference(model_list, results_dir, n_samples=256, vertical_sub_plots=False):

    configure_plt()

    if not os.path.isdir(results_dir + '/plots'):
        os.makedirs(results_dir + '/plots', exist_ok=True)

    avg_r2_list = np.zeros(len(model_list))
    avg_l1_list = np.zeros(len(model_list))

    if vertical_sub_plots:
        v_plots = 2
        h_plots = 1
        figsize = (12, 16)
        bbox = (0.5, -0.4)
        ncol = 3
    else:
        v_plots = 1
        h_plots = 2
        figsize = (34, 8)
        bbox = (-0.1, -0.3)
        ncol = 7

    r2_fig, r2_ax = plt.subplots(v_plots, h_plots, figsize=figsize)
    l1_fig, l1_ax = plt.subplots(v_plots, h_plots, figsize=figsize)

    for i, model_name in enumerate(model_list):

        if model_name == 'ours_savp':
            labels = 'savp'
        elif model_name == 'ours_vae':
            labels = 'savp_vae'
        else:
            labels = model_name

        inference_metrics_dir = os.path.join(results_dir, labels, 'inference_metrics.pickle')

        if model_name == 'svg':
            labels = 'svg-lp'

        labels = labels.replace("_", "-")

        # == R2
        with open(inference_metrics_dir, 'rb') as f:
            abs_diff_by_step_mean, abs_diff_by_step_std, avg_r2_by_step_x, avg_r2_by_step_y, avg_r2, avg_l1 = pickle.load(
                f)

        avg_r2_list[i] = avg_r2
        avg_l1_list[i] = avg_l1

        plt.sca(r2_ax[0])
        plt.plot(range(3, 27 + 3 - 1, 2), avg_r2_by_step_x[1::2], label=labels.upper(), color=color_mean[i],
                 linewidth=LINEWIDTH)
        plt.scatter(range(3, 27 + 3 - 1, 2), avg_r2_by_step_x[1::2], color=color_mean[i], s=60)
        plt.plot(range(2, 27 + 2, 2), avg_r2_by_step_x[::2], linestyle='dashed', color=color_mean[i],
                 linewidth=LINEWIDTH)
        plt.scatter(range(2, 27 + 2, 2), avg_r2_by_step_x[::2], color=color_mean[i], s=60)

        plt.sca(r2_ax[1])
        plt.plot(range(3, 27 + 3 - 1, 2), avg_r2_by_step_y[1::2], label=labels.upper(), color=color_mean[i],
                 linewidth=LINEWIDTH)
        plt.scatter(range(3, 27 + 3 - 1, 2), avg_r2_by_step_y[1::2], color=color_mean[i], s=60)
        plt.plot(range(2, 27 + 2, 2), avg_r2_by_step_y[::2], linestyle='dashed', color=color_mean[i],
                 linewidth=LINEWIDTH)
        plt.scatter(range(2, 27 + 2, 2), avg_r2_by_step_y[::2], color=color_mean[i], s=60)

        # == L1
        plt.sca(l1_ax[0])
        plot_mean_and_CI(abs_diff_by_step_mean[1::2, 0],
                         abs_diff_by_step_std[1::2, 0],
                         n_samples, color_mean[i], color_shading[i], labels.upper(), start=3, end=29, step=2)

        plot_mean_and_CI(abs_diff_by_step_mean[::2, 0],
                         abs_diff_by_step_std[::2, 0],
                         n_samples, color_mean[i], color_shading[i], start=2, end=29, step=2, linestyle='dashed')

        plt.sca(l1_ax[1])
        plot_mean_and_CI(abs_diff_by_step_mean[1::2, 1],
                         abs_diff_by_step_std[1::2, 1],
                         n_samples, color_mean[i], color_shading[i], labels.upper(), start=3, end=29, step=2)

        plot_mean_and_CI(abs_diff_by_step_mean[::2, 1],
                         abs_diff_by_step_std[::2, 1],
                         n_samples, color_mean[i], color_shading[i], start=2, end=29, step=2, linestyle='dashed')

    # ===== R2 L1 table
    r2_string = [str(avg_r2_list[i]) for i in range(len(model_list))]
    l1_string = [str(avg_l1_list[i]) for i in range(len(model_list))]

    table_fig = plt.figure(figsize=(10, 15))
    plt.figure(table_fig.number)

    collabel = ['Model', 'R2 score', 'L1 score']
    plt.axis('tight')
    plt.axis('off')
    names = model_list

    names = names[:len(model_list)]  # assuming model_list has the order above

    the_table = plt.table(
         cellText=np.stack([np.array(names), np.array(r2_string), np.array(l1_string)], axis=0).transpose(),
         colLabels=collabel,
         loc='center',
         cellLoc='center',
         colWidths=[0.5, 0.5, 0.5])

    plt.savefig(os.path.join(results_dir, 'plots', 'r2_l1.eps'), format='eps')
    plt.savefig(os.path.join(results_dir, 'plots', 'r2_l1.png'), format='png')

    # ===== R2
    plt.sca(r2_ax[0])
    plt.title(r'Average $R^{2}$ on the x axis')
    plt.xlabel('Time Step')
    plt.ylabel('Average $R^{2}$')
    plt.ylim(-0.2, 1)
    plt.xticks(range(0, 31, 2))
    plt.yticks(np.arange(-0.2, 1.0, 0.2))
    plt.axvline(x=12, zorder=-1, color='darkblue')
    plt.axhline(y=0, zorder=-1, color='black', linewidth=0.7)
    plt.grid()

    plt.sca(r2_ax[1])
    plt.title(r'Average $R^{2}$ on the y axis')
    plt.xlabel('Time Step')
    plt.ylabel('Average $R^{2}$')
    plt.ylim(-0.2, 1)
    plt.xticks(range(0, 31, 2))
    plt.yticks(np.arange(-0.2, 1.0, 0.2))
    plt.axvline(x=12, zorder=-1, color='darkblue')
    plt.axhline(y=0, zorder=-1, color='black', linewidth=0.7)
    plt.plot(range(1), -10, color='black', linestyle='solid', label='odd time steps', linewidth=LINEWIDTH)
    plt.plot(range(1), -10, color='black', linestyle='dashed', label='even time steps', linewidth=LINEWIDTH)
    plt.grid()

    plt.figure(r2_fig.number)
    lgd = plt.legend(shadow=True, fancybox=True, loc='lower center',
                     bbox_to_anchor=bbox, ncol=ncol)
    plt.subplots_adjust(hspace=0.25)
    plt.savefig(os.path.join(results_dir, 'plots', 'r2.eps'), format='eps',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'plots', 'r2.png'), format='png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')

    # ===== L1
    plt.sca(l1_ax[0])
    plt.title(r'Average MAE on the x axis')
    plt.xlabel('Time Step')
    plt.ylabel('Average MAE')
    plt.xticks(range(0, 31, 2))
    plt.axvline(x=12, zorder=-1, color='darkblue')
    plt.grid()
    plt.ylim(0.0, None)

    plt.sca(l1_ax[1])
    plt.title(r'Average MAE on the y axis')
    plt.xlabel('Time Step')
    plt.ylabel('Average MAE')
    plt.xticks(range(0, 31, 2))
    plt.axvline(x=12, zorder=-1, color='darkblue')
    plt.grid()
    plt.ylim(0.0, None)
    plt.plot(range(1), -10, color='black', linestyle='solid', label='odd time steps', linewidth=LINEWIDTH)
    plt.plot(range(1), -10, color='black', linestyle='dashed', label='even time steps', linewidth=LINEWIDTH)

    plt.figure(l1_fig.number)
    lgd = plt.legend(shadow=True, fancybox=True, loc='lower center',
                     bbox_to_anchor=bbox, ncol=ncol)
    plt.subplots_adjust(hspace=0.25)
    plt.savefig(os.path.join(results_dir, 'plots', 'l1.eps'), format='eps',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'plots', 'l1.png'), format='png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')


def make_plots_vp_metrics(model_list, results_dir, n_samples=256, vertical_sub_plots=False):

    configure_plt()

    if not os.path.isdir(results_dir + '/plots'):
        os.makedirs(results_dir + '/plots', exist_ok=True)

    if vertical_sub_plots:
        v_plots = 2
        h_plots = 1
        figsize = (12, 12)
        bbox = (0.5, -0.4)
        ncol = 3
    else:
        v_plots = 1
        h_plots = 2
        figsize = (34, 8)
        bbox = (-0.1, -0.3)
        ncol = 7

    fvd_list = np.zeros(len(model_list))
    psnr_ssim_fig, psnr_ssim_ax = plt.subplots(v_plots, h_plots, figsize=figsize)

    for i, model_name in enumerate(model_list):

        if model_name == 'ours_savp':
            labels = 'savp'
        elif model_name == 'ours_vae':
            labels = 'savp_vae'
        else:
            labels = model_name

        metrics_dir = os.path.join(results_dir, labels, 'metrics.pickle')

        if model_name == 'svg':
            labels = 'svg-lp'

        labels = labels.replace("_", "-")

        # ===== existing metrics
        with open(metrics_dir, 'rb') as f:
            psnr_mean, psnr_standard_dev, ssim_mean, ssim_standard_dev, fvd_val = pickle.load(f)

        # plot PSNR the data
        plt.sca(psnr_ssim_ax[0])
        plot_mean_and_CI(psnr_mean, psnr_standard_dev, n_samples, color_mean[i], color_shading[i], labels.upper())

        # plot SSIM the data
        plt.sca(psnr_ssim_ax[1])
        plot_mean_and_CI(ssim_mean, ssim_standard_dev, n_samples, color_mean[i], color_shading[i], labels.upper())

        fvd_list[i] = fvd_val

    # ===== FVD table
    fvd_string = [str(fvd_list[i]) for i in range(len(model_list))]

    table_fig = plt.figure(figsize=(10, 15))
    plt.figure(table_fig.number)

    collabel = ['Model', 'FVD Score']
    plt.axis('tight')
    plt.axis('off')
    names = model_list

    names = names[:len(model_list)]  # assuming model_list has the order above

    the_table = plt.table(cellText=np.stack([np.array(names), np.array(fvd_string)], axis=0).transpose(),
                          colLabels=collabel,
                          loc='center',
                          cellLoc='center',
                          colWidths=[0.5, 0.5])

    plt.savefig(os.path.join(results_dir, 'plots', 'fvd.eps'), format='eps')
    plt.savefig(os.path.join(results_dir, 'plots', 'fvd.png'), format='png')

    # ===== PSNR & SSIM
    plt.sca(psnr_ssim_ax[0])
    plt.title('Average PSNR by prediction step')
    plt.xlabel('Time Step')
    plt.ylabel('Average PSNR')
    plt.xticks(range(0, 31, 2))
    plt.grid()
    plt.axvline(x=12, zorder=-1, color='darkblue')

    plt.sca(psnr_ssim_ax[1])
    plt.title('Average SSIM by prediction step')
    plt.xlabel('Time Step')
    plt.ylabel('Average SSIM')
    plt.xticks(range(0, 31, 2))
    plt.grid()
    lgd = plt.legend(shadow=True, fancybox=True, loc='lower center',
                     bbox_to_anchor=bbox, ncol=ncol)
    plt.axvline(x=12, zorder=-1, color='darkblue')

    plt.figure(psnr_ssim_fig.number)
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(os.path.join(results_dir, 'plots', 'psnr_ssim.eps'), format='eps',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'plots', 'psnr_ssim.png'), format='png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')


def make_plots_regressions(model_list, results_dir, inference_models_dir, n_samples=256, vertical_sub_plots=False):


    FLAGS.batch_size = 1
    FLAGS.train_val_split = 0.0
    FLAGS.sequence_length_test = 28
    sequence_length = FLAGS.sequence_length_test

    if vertical_sub_plots:
        v_plots = 2
        h_plots = 1
        figsize = (12, 12)
        bbox = (0.5, -0.4)
        ncol = 3
    else:
        v_plots = 1
        h_plots = 2
        figsize = (34, 8)
        bbox = (-0.1, -0.3)
        ncol = 7

    configure_plt()

    all_pred_seq = {name: np.zeros([n_samples, sequence_length - 1, 2]) for name in model_list}
    all_gt_seq = {name: np.zeros([n_samples, sequence_length - 1, 2]) for name in model_list}
    all_r2_scores_x = {name: [0] * n_samples for name in model_list}
    all_r2_scores_y = {name: [0] * n_samples for name in model_list}

    x_plot_fig, x_plot_ax = plt.subplots(v_plots, h_plots, figsize=figsize)
    y_plot_fig, y_plot_ax = plt.subplots(v_plots, h_plots, figsize=figsize)

    for model_name in model_list:

        K.clear_session()
        print('Model: ', model_name)

        processed_data_dir = None if model_name is not 'svg' else os.path.join(os.path.expanduser('~/'),
                                                                               'data/processed_data')
        d = BairDataReader(dataset_dir=FLAGS.bair_dir,
                           processed_dataset_dir=processed_data_dir,
                           batch_size=1,
                           sequence_length_train=30,
                           sequence_length_test=30,
                           shuffle=True,
                           dataset_repeat=None)

        test_iterator = d.build_tfrecord_iterator(mode='test')
        inputs = test_iterator.get_next()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # ===== prepare image pairs for input
        seq_len = inputs['images'].shape[1] - 1
        images_t = tf.slice(inputs['images'], (0, 0, 0, 0, 0), (-1, seq_len, 64, 64, 3))
        images_tp1 = tf.slice(inputs['images'], (0, 1, 0, 0, 0), (-1, seq_len, 64, 64, 3))
        test_image_pairs = tf.concat([images_t, images_tp1], axis=-1)

        # ===== restore the trained inference
        model_path = os.path.join(inference_models_dir, model_name)
        weight_path = os.path.join(model_path, 'model_weights.h5')

        model_input = Input(shape=(sequence_length - 1, 64, 64, 6))
        inferred_actions = action_inference_model(model_input)

        model = Model(inputs=model_input, outputs=inferred_actions)
        model.load_weights(weight_path)

        # ===== forward pass on test set
        for i in range(n_samples):
            try:
                imgs_gt, deltas_xy_gt = sess.run([test_image_pairs, inputs['action_targets']])
                deltas_xy_pred = model.predict(imgs_gt[:, -(sequence_length - 1):], steps=1)

                deltas_xy_gt = deltas_xy_gt[:, -(sequence_length - 1):]

                all_pred_seq[model_name][i] = deltas_xy_pred
                all_gt_seq[model_name][i] = deltas_xy_gt[0]

                # get r2 scores for every sequence
                all_r2_scores_x[model_name][i] = r2_score(y_true=deltas_xy_gt[0, :, 0],
                                                          y_pred=deltas_xy_pred[:, 0])
                all_r2_scores_y[model_name][i] = r2_score(y_true=deltas_xy_gt[0, :, 1],
                                                          y_pred=deltas_xy_pred[:, 1])

            except tf.errors.OutOfRangeError:
                print('end of dataset. %d iterations' % i)
                break

    # find best score on x
    plt.sca(x_plot_ax[0])
    find_matching_seqs_and_plot('best', 0, all_gt_seq, all_pred_seq, all_r2_scores_x, model_list)

    plt.title(r'Best example of inferred actions in the x axis')
    plt.xlabel('Time Step')
    plt.ylabel('$\Delta x$')
    plt.xticks(range(0, 31, 2))

    # find worst score on x
    plt.sca(x_plot_ax[1])
    find_matching_seqs_and_plot('worst', 0, all_gt_seq, all_pred_seq, all_r2_scores_x, model_list)

    plt.title(r'Worst example of inferred actions in the x axis')
    plt.xlabel('Time Step')
    plt.ylabel('$\Delta x$')
    plt.xticks(range(0, 31, 2))

    # save
    plt.figure(x_plot_fig.number)
    lgd = plt.legend(shadow=True, fancybox=True, loc='lower center',
                     bbox_to_anchor=bbox, ncol=ncol)
    plt.subplots_adjust(hspace=0.30)
    plt.savefig(os.path.join(results_dir, 'plots', 'x_reg.eps'), format='eps',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'plots', 'x_reg.png'), format='png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')

    # find best score on y
    plt.sca(y_plot_ax[0])
    find_matching_seqs_and_plot('best', 1, all_gt_seq, all_pred_seq, all_r2_scores_y, model_list)

    plt.title(r'Best example of inferred actions in the y axis')
    plt.xlabel('Time Step')
    plt.ylabel('$\Delta y$')
    plt.xticks(range(0, 31, 2))

    # find worst score on y
    plt.sca(y_plot_ax[1])
    find_matching_seqs_and_plot('worst', 1, all_gt_seq, all_pred_seq, all_r2_scores_y, model_list)

    plt.title(r'Worst example of inferred actions in the y axis')
    plt.xlabel('Time Step')
    plt.ylabel('$\Delta y$')
    plt.xticks(range(0, 31, 2))

    # save
    plt.figure(y_plot_fig.number)
    lgd = plt.legend(shadow=True, fancybox=True, loc='lower center',
                     bbox_to_anchor=bbox, ncol=ncol)
    plt.subplots_adjust(hspace=0.30)
    plt.savefig(os.path.join(results_dir, 'plots', 'y_reg.eps'), format='eps',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'plots', 'y_reg.png'), format='png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')


def find_matching_seqs_and_plot(mode, dim, all_gt_seq, all_pred_seq, all_r2_scores, model_list):
    if mode == 'best':
        name_found = max(all_r2_scores, key=all_r2_scores.get)  # model with the best score
        seq_n = np.argmax(all_r2_scores[name_found])  # sequence number that produced the best score
        print('Best: ', name_found)
    elif mode == 'worst':
        name_found = min(all_r2_scores, key=all_r2_scores.get)  # model with the best score
        seq_n = np.argmin(all_r2_scores[name_found])  # sequence number that produced the best score
        print('Worst: ', name_found)

    found_seq = all_gt_seq[name_found][seq_n]

    # plot best/worst
    plot_sequences(dim, all_gt_seq[name_found][seq_n], all_pred_seq[name_found][seq_n], name_found, linewidth=4.0)

    # plot the others for the same sequence
    for name in model_list:
        if name == name_found:
            continue
        index = np.where(all_gt_seq[name] == found_seq)
        # print('Index: ', index)
        # print('Index: ', index[0][0])
        plot_sequences(dim, all_gt_seq[name][index[0][0]], all_pred_seq[name][index[0][0]], name, linewidth=4.0)

    # plot the ground truth
    plt.plot(range(2, 29), found_seq[:, dim], color='black', label='Ground Truth', linewidth=3.0, linestyle='dashed',
             zorder=3)
    plt.scatter(range(2, 29), found_seq[:, dim], color='black', zorder=3)
    plt.axhline(y=0, color='black', linewidth=1.0)
    plt.axvline(x=12)


def plot_sequences(dim, y_true, all_pred, name, start=2, end=29, linewidth=3.0, dot_size=60):

    color = None

    if name == 'savp_vae':
        color = 'orangered'
    elif name == 'savp':
        color = 'mediumvioletred'
    elif name == 'sv2p':
        color = 'teal'
    elif name == 'cdna':
        color = 'gold'
    elif name == 'svg':
        color = 'mediumblue'

    label = name.upper().replace("_", "-")
    plt.plot(range(start, end), all_pred[:, dim], color=color, label=label, linewidth=linewidth)
    plt.scatter(range(start, end), all_pred[:, dim], color=color, s=dot_size)

