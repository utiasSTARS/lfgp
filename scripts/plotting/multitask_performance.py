import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob

import utils as plot_utils
from data_locations import main_performance as data_locations
import common as plot_common


opts = plot_utils.get_default_opts()
#### Options ########################################################################################################
# fig
opts['font_size'] = 24
opts['data_locations'] = data_locations
opts['aux_names'] = [
    ['Open', 'Close', 'Stack', 'Lift', 'Reach', 'Move'],
    ['Open', 'Close', 'Unstack-Stack', 'Lift', 'Reach', 'Move'],
    ['Open', 'Close', 'Bring', 'Lift', 'Reach', 'Move'],
    ['Open', 'Close', 'Insert', 'Bring', 'Lift', 'Reach', 'Move']
]
opts['aux_orders'] = [
    [2, 0, 1, 3, 4, 5],
    [2, 0, 1, 3, 4, 5],
    [2, 0, 1, 3, 4, 5],
    [2, 0, 1, 3, 4, 5, 6],
]
opts['algo_dir_names'] = ['dac-x', 'multitask_dac', 'multitask_bc']
opts['algo_titles'] = ['LfGP (multi)', 'LfGP-NS (multi)', 'BC (multi)']
opts['fig_path'] = opts['root_dir'] + "/figures/multitask_performance"
opts['valid_task'] = [True, True, True, True]
# opts['valid_task'] = [False, True, False, False]
# opts['rl_eval_eps_per_task'] = 10  
# opts['bc_eval_eps_per_task'] = 10
# opts['eval_interval'] = 10000

root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
    plot_common.get_path_defaults(fig_name="multitask_performance")

task_dir_names, valid_task, task_titles, main_task_i, num_aux, task_data_filenames, num_eval_steps_to_use = \
    plot_common.get_task_defaults()

algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task = plot_common.get_algo_defaults()

fig_shape, plot_size, num_stds, font_size, eval_interval, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
    include_expert_baseline = plot_common.get_fig_defaults()

aux_names = [
    ['Open', 'Close', 'Stack', 'Lift', 'Reach', 'Move'],
    ['Open', 'Close', 'Unstack-Stack', 'Lift', 'Reach', 'Move'],
    ['Open', 'Close', 'Bring', 'Lift', 'Reach', 'Move'],
    ['Open', 'Close', 'Insert', 'Bring', 'Lift', 'Reach', 'Move']
]
aux_orders = [
    [2, 0, 1, 3, 4, 5],
    [2, 0, 1, 3, 4, 5],
    [2, 0, 1, 3, 4, 5],
    [2, 0, 1, 3, 4, 5, 6],
]
#####################################################################################################################

# pretty plotting, allow tex
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

all_successes, all_returns = plot_utils.get_returns_successes("multitask_performance", data_locations)

# dicts are all_successes['task']['algo']['raw/mean/std'],
# raw shape: (seed, timestep, aux task, eval ep)
# mean and std shape: (timestep, aux_task)

# fig 1: success rate of each aux while executing own task
# own_task_s_figs = [plt.subplots(nrows=1, ncols=opts['num_aux[task_i])
nrows = len(task_dir_names)
ncols = max(num_aux)
own_task_s_fig = plt.figure(figsize=[plot_size[0] * ncols, plot_size[1] * nrows])
own_task_r_fig = plt.figure(figsize=[plot_size[0] * ncols, plot_size[1] * nrows])


for task_i, task in enumerate(task_dir_names):
    if not valid_task[task_i]: 
        print(f"Task {task} set to false in valid_task, skipping in plotting")
        continue

    # for aux_i in range(num_aux[task_i]):
    for col_i, aux_i in enumerate(aux_orders[task_i]):
        # plt_index = task_i * ncols + aux_i + 1
        plt_index = task_i * ncols + col_i + 1
        for plot_type, fig, data in zip(['s', 'r'], [own_task_s_fig, own_task_r_fig], [all_successes, all_returns]):
            ax = fig.add_subplot(nrows, ncols, plt_index)
            ax.set_title(aux_names[task_i][aux_i], fontsize=font_size)

            # if aux_i == 0:
            if col_i == 0:
                ax.set_ylabel(task_titles[task_i], fontsize=font_size)

            for algo_i, algo in enumerate(algo_dir_names):
                if algo in multitask_algos or aux_i == 2:
                    plot_utils.plot_mean_std(ax, aux_i, algo, algo_i, data[task][algo],
                                            algo_label=algo_titles[algo_i] if (task_i == 0 and aux_i == 2) else None)
            
            # pretty
            if plot_type == 's':
                ax.set_ylim(-.01, 1.05)
                ax.set_yticks([0, .5, 1])
                ax.set_yticks([0,.25, .5, .75, 1], minor=True)
            ax.tick_params(labelsize=font_size - 4)
            if task == 'insert_0':
                ax.set_xlim(0, 4.1)
                ax.set_xticks([1, 2, 3, 4])
                ax.set_xticks(np.arange(0, 4, 0.5), minor=True)
                ax.grid(which='both', alpha=0.5)
            else:
                ax.set_xlim(0, 2.1)
                ax.set_xticks([0.5, 1, 1.5, 2])
                # ax.set_xticks([0,1,2,3,4], minor=True)
                ax.grid(which='both', alpha=0.5)


for fig, fig_name in zip([own_task_s_fig, own_task_r_fig], ['s_fig.pdf', 'r_fig.pdf']):
    fig.tight_layout()
    fig.legend(fancybox=True, shadow=True, fontsize=font_size, loc="right",
               bbox_to_anchor=(0.98, 0.5))
            #    ncol=len(algo_dir_names) + 1, bbox_to_anchor=(0.5, -0.31))

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("Updates/steps (millions)", fontsize=font_size + 4, labelpad=10)
    
    if 's_fig' in fig_name:
        ax.set_ylabel("Success Rate", fontsize=font_size + 4, labelpad=32)
    else:
        ax.set_ylabel("Episode Return", fontsize=font_size + 4, labelpad=30)

    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

# fig 2: success rate of each aux while running the main task



