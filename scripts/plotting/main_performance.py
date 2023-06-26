import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import math

from data_locations import main_performance as data_locations
import common as plot_common


#### Options ########################################################################################################
root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
    plot_common.get_path_defaults(fig_name="main_performance")

task_dir_names, valid_task, task_titles, main_task_i, num_aux, task_data_filenames, num_eval_steps_to_use = \
    plot_common.get_task_defaults()

algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task = plot_common.get_algo_defaults()

fig_shape, plot_size, num_stds, font_size, eval_interval, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
    include_expert_baseline = plot_common.get_fig_defaults()

side_legend = True
#####################################################################################################################

# pretty plotting, allow tex
# plt.rcParams.update({"font.family": "serif", "font.serif": "Times", "text.usetex": True, "pgf.rcfonts": False})
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

s_fig, s_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])
r_fig, r_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                             figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])
s_axes_flat = s_axes.flatten()
r_axes_flat = r_axes.flatten()

# get returns and successes
all_returns = dict.fromkeys(task_dir_names)
all_successes = dict.fromkeys(task_dir_names)
for task_i, task in enumerate(task_dir_names):
    if not valid_task[task_i]:
        print(f"Task {task} set to false in valid_task, skipping")
        continue
    all_successes[task] = { algo : dict(raw=[], mean=[], std=[]) for algo in algo_dir_names }
    all_returns[task] = { algo : dict(raw=[], mean=[], std=[]) for algo in algo_dir_names }
    for algo_i, algo in enumerate(algo_dir_names):

        # folder structure is task/seed/algo/experiment_name/datetime
        algo_dir, experiment_name = data_locations[task][algo].split('/')

        data_path = os.path.join(experiment_root_dir, task, '1', algo_dir)
        if not os.path.exists(data_path):
            print("No path found at %s for task %s algo %s, moving on in data cleaning" % (data_path, task, algo))
            continue
        for seed in seeds:
            # data_path = os.path.join(root_dir, top_task_dirs[task_i],  seed, algo, experiment_name)
            data_path = os.path.join(experiment_root_dir, task,  seed, algo_dir, experiment_name)

            # find datetime folder
            try:
                dirs = sorted([os.path.join(data_path, found) for found in os.listdir(data_path)
                              if os.path.isdir(os.path.join(data_path, found))])
                if len(dirs) > 1:
                    print(f"WARNING: multiple folders found at {data_path}, using {dirs[-1]}")
                data_path = dirs[-1]
            except:
                print(f"Error at data_path {data_path}")
                # import ipdb; ipdb.set_trace()

            if not os.path.exists(data_path):
                print("No path found at %s for task %s, moving on in data cleaning" % (data_path, task))
                continue

            data_file = os.path.join(data_path, task_data_filenames[task_i])
            if not os.path.isfile(data_file):
                data_file = os.path.join(data_path, 'train.pkl')  # default for cases where we didn't want to rename

            data = pickle.load(open(data_file, 'rb'))

            suc_data = np.array(data['evaluation_successes_all_tasks']).squeeze()
            ret_data = np.array(data['evaluation_returns']).squeeze()

            # adjust arrays to compensate for recycle scheduler
            if algo in multitask_algos:
                suc_fixed = []
                ret_fixed = []

                for aux_i in range(num_aux[task_i]):
                    rets_slice = slice(aux_i * eval_eps_per_task, aux_i * eval_eps_per_task + eval_eps_per_task)
                    # for all algos, -1 index is episode, -2 is which aux, 0 is eval step index
                    suc_fixed.append(suc_data[..., aux_i, rets_slice])
                    ret_fixed.append(ret_data[..., aux_i, rets_slice])

                suc_data = np.array(suc_fixed)
                ret_data = np.array(ret_fixed)
                # if not 'bc' in algo:
                #     suc_data = np.swapaxes(suc_data, 0, 1)
                #     ret_data = np.swapaxes(ret_data, 0, 1)

                # now order is eval step index, aux index, eval ep index
                suc_data = np.swapaxes(suc_data, 0, 1)
                ret_data = np.swapaxes(ret_data, 0, 1)

            # remove extra eval step indices if there are any
            suc_data = suc_data[..., :num_eval_steps_to_use[task_i], :]
            ret_data = ret_data[..., :num_eval_steps_to_use[task_i], :]

            all_successes[task][algo]['raw'].append(suc_data)
            all_returns[task][algo]['raw'].append(ret_data)

        all_returns[task][algo]['raw'] = np.array(all_returns[task][algo]['raw']).squeeze()
        all_successes[task][algo]['raw'] = np.array(all_successes[task][algo]['raw']).squeeze()

        # take along episode axis, then along seed axis..matters for std
        for dat in [all_returns[task][algo], all_successes[task][algo]]:
            dat['mean'] = dat['raw'].mean(axis=-1).mean(axis=0)
            dat['std'] = dat['raw'][:num_eval_steps_to_use[task_i]].mean(axis=-1).std(axis=0)

skipped_tasks = 0
# plotting -- for now only main task
for task_i, task in enumerate(task_dir_names):
    if not valid_task[task_i]:
        print(f"Task {task} set to false in valid_task, skipping in plotting")
        skipped_tasks += 1
        continue
    ax_i = task_i - skipped_tasks
    s_ax = s_axes_flat[ax_i]
    r_ax = r_axes_flat[ax_i]
    for algo_i, algo in enumerate(algo_dir_names):
        if algo not in all_successes[task].keys():
            print(f"No data for algo {algo} and task {task}, skipping")
            continue
        for ax, task_algo_data in zip([s_ax, r_ax], [all_successes[task][algo], all_returns[task][algo]]):
            if algo in multitask_algos:
                mean = task_algo_data['mean'][..., main_task_i[task_i]]
                std = task_algo_data['std'][..., main_task_i[task_i]]
            else:
                mean = task_algo_data['mean']
                std = task_algo_data['std']

            if algo in multitask_algos:
                line_style = '-'
            else:
                line_style = '--'

            x_vals = np.array(range(eval_interval, eval_interval * len(mean) + 1, eval_interval * subsample_rate))
            x_vals = x_vals / x_val_scale

            # mean = mean[::subsample_rate]
            # std = std[::subsample_rate]

            # test combining data at subsample rate to see if it reduces noise
            if subsample_rate > 1:
                new_mean = []
                new_std = []
                for samp_start in range(0, task_algo_data['raw'].shape[1], subsample_rate):
                    new_samp = task_algo_data['raw'][:, samp_start:samp_start+subsample_rate, ...]
                    if algo in multitask_algos:
                        new_mean.append(new_samp.mean(axis=(1, -1))[:, main_task_i[task_i]].mean())
                        new_std.append(new_samp.mean(axis=(1, -1))[:, main_task_i[task_i]].std())
                    else:
                        new_mean.append(new_samp.mean(axis=(1, -1)).mean())
                        new_std.append(new_samp.mean(axis=(1, -1)).std())
                mean = np.array(new_mean)
                std = np.array(new_std)

            ax.plot(x_vals, mean, label=algo_titles[algo_i] if task_i == 1 else "",
                    color=cmap(algo_i), linewidth=linewidth, linestyle=line_style)

            try:
                ax.fill_between(x_vals, mean - num_stds * std, mean + num_stds * std, facecolor=cmap(algo_i),
                                alpha=std_alpha)
            except:
                import ipdb; ipdb.set_trace()

    # pretty up plot, add expert baselines
    for plot_type, ax in zip(['s', 'r'], [s_ax, r_ax]):
        # baselines
        if include_expert_baseline:
            exp_data = pickle.load(open(expert_perf_files[task_i], 'rb'))
            if plot_type == 's':
                if task == 'unstack_stack_env_only_0':
                    perf_mean = exp_data['executed_task_successes'].mean()  # this one is broken for some reason
                else:
                    perf_mean = exp_data['executed_task_successes'][expert_perf_file_main_task_i[task_i]].mean()

            else:
                if task == 'unstack_stack_env_only_0':
                    perf_mean = exp_data['executed_task_returns'].mean()  # this one is broken for some reason
                else:
                    perf_mean = exp_data['executed_task_returns'][expert_perf_file_main_task_i[task_i]].mean()

            ax.axhline(perf_mean, label="Expert" if task_i == 1 else "", color="k",
                    linewidth=linewidth, linestyle='-.')

        # pretty
        ax.grid(alpha=0.5)
        ax.set_title(task_titles[task_i], fontsize=font_size)
        if task == 'insert_0':
            from matplotlib.ticker import AutoMinorLocator
            minor_locator = AutoMinorLocator(2)
            ax.set_xticks(np.arange(1.0, 4.1, 1.0))
            ax.xaxis.set_minor_locator(minor_locator)
            ax.grid(alpha=0.5, which='both')
        # else:    
        #     ax.set_xticks(np.arange(1.0, 2.1, 1.0))
        #     ax.xaxis.grid(True, which='minor')
        if plot_type == 's':
            ax.set_ylim(-.01, 1.01)

# plt.tight_layout()  # careful with this...only applies to one fig and doesn't seem to work right
for fig, fig_name in zip([s_fig, r_fig], ['s_fig.pdf', 'r_fig.pdf']):

    if side_legend:
        fig_name = "sideleg_" + fig_name

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # ax.set_xlabel("Environment Steps (millions)", fontsize=font_size)
    ax.set_xlabel("Updates/steps (millions)", fontsize=font_size)
    # ax.xaxis.set_label_coords(.57, -.15)  # if we have the 10^6 scientific notation
    if 's_fig' in fig_name:
        ax.set_ylabel("Success Rate", fontsize=font_size)
    else:
        ax.set_ylabel("Episode Return", fontsize=font_size)
    
    if side_legend:
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center", ncol=1, bbox_to_anchor=(0.98, 0.15)) 
    else:
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-2, loc="lower center",
                ncol=int(math.ceil((len(algo_dir_names) + 1))), 
                bbox_to_anchor=(0.5, -0.3))    

    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')