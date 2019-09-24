import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 20
MEDIUM_SIZE =24
BIGGER_SIZE = 40
LINEWIDTH = 4.0


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


def make_plots_action_inference(model_list, results_dir, n_samples=256):

    configure_plt()

    color_mean = ['orangered', 'purple', 'teal', 'gold']
    color_shading = ['lightpink', 'thistle', 'paleturquoise', 'lemonchiffon']

    if not os.path.isdir(results_dir + '/plots'):
        os.makedirs(results_dir + '/plots', exist_ok=True)

    avg_r2_list = np.zeros(len(model_list))
    avg_l1_list = np.zeros(len(model_list))

    r2_fig, r2_ax = plt.subplots(2, 1, figsize=(12, 16))
    l1_fig, l1_ax = plt.subplots(2, 1, figsize=(12, 16))

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
        plt.plot(range(2, 27 + 2, 2), avg_r2_by_step_x[::2], linestyle='dashed', color=color_mean[i],
                 linewidth=LINEWIDTH)

        plt.sca(r2_ax[1])
        plt.plot(range(3, 27 + 3 - 1, 2), avg_r2_by_step_y[1::2], label=labels.upper(), color=color_mean[i],
                 linewidth=LINEWIDTH)
        plt.plot(range(2, 27 + 2, 2), avg_r2_by_step_y[::2], linestyle='dashed', color=color_mean[i],
                 linewidth=LINEWIDTH)

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
    # r2_string = [str(avg_r2_list[i]) for i in range(len(model_list))]
    # l1_string = [str(avg_l1_list[i]) for i in range(len(model_list))]
    #
    # table_fig = plt.figure(figsize=(10, 15))
    # plt.figure(table_fig.number)
    #
    # collabel = ['Model', 'R2 score', 'L1 score']
    # plt.axis('tight')
    # plt.axis('off')
    # names = ['CDNA', 'SAVP', 'SAVP-VAE', 'SVG']
    #
    # names = names[:len(model_list)]  # assuming model_list has the order above
    #
    # the_table = plt.table(
    #     cellText=np.stack([np.array(names), np.array(r2_string), np.array(l1_string)], axis=0).transpose(),
    #     colLabels=collabel,
    #     loc='center',
    #     cellLoc='center',
    #     colWidths=[0.5, 0.5, 0.5])
    #
    # plt.savefig(os.path.join(results_dir, 'plots', 'r2_l1.eps'), format='eps')
    # plt.savefig(os.path.join(results_dir, 'plots', 'r2_l1.png'), format='png')

    # ===== R2
    plt.sca(r2_ax[0])
    plt.title(r'Average $R^{2}$ on the x axis')
    plt.xlabel('Time Step')
    plt.ylabel('Average $R^{2}$')
    plt.ylim(-1.6, 1)
    plt.xticks(range(0, 31, 2))
    plt.yticks(np.arange(-1.5, 1.0, 0.2))
    plt.axvline(x=12, zorder=-1, color='darkblue')
    plt.axhline(y=0, zorder=-1, color='black', linewidth=0.7)
    plt.grid()

    plt.sca(r2_ax[1])
    plt.title(r'Average $R^{2}$ on the y axis')
    plt.xlabel('Time Step')
    plt.ylabel('Average $R^{2}$')
    plt.ylim(-1.6, 1)
    plt.xticks(range(0, 31, 2))
    plt.yticks(np.arange(-1.5, 1.0, 0.2))
    plt.axvline(x=12, zorder=-1, color='darkblue')
    plt.axhline(y=0, zorder=-1, color='black', linewidth=0.7)
    plt.plot(range(1), -10, color='black', linestyle='solid', label='odd time steps', linewidth=LINEWIDTH)
    plt.plot(range(1), -10, color='black', linestyle='dashed', label='even time steps', linewidth=LINEWIDTH)
    plt.grid()

    plt.figure(r2_fig.number)
    lgd = plt.legend(shadow=True, fancybox=True, loc='lower center',
                     bbox_to_anchor=(0.5, -0.4), ncol=3)
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
                     bbox_to_anchor=(0.5, -0.4), ncol=3)
    plt.subplots_adjust(hspace=0.25)
    plt.savefig(os.path.join(results_dir, 'plots', 'l1.eps'), format='eps',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'plots', 'l1.png'), format='png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
