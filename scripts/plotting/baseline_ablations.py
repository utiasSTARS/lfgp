from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import math

from data_locations import baseline_ablations as data_locations
import common as plot_common


#### Options ########################################################################################################
root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
    plot_common.get_path_defaults(fig_name="baseline_ablations", task_inds=[0])  # stack only

task_dir_names, valid_task, task_titles, main_task_i, num_aux, task_data_filenames, num_eval_steps_to_use = \
    plot_common.get_task_defaults(task_inds=[0])

algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task = plot_common.get_algo_defaults()
num_datas = ["bc_dac_alternatives"]  # keys in dictionary
num_datas_labels = ["BC/DAC Alternatives"]
algo_dir_names = [
    ['multitask_bc', 'multitask_bc_valid', 'dac', 'gail', 'bc', 'bc_valid'],
]
algo_titles = [
    ['BC (multi)', 'BC (multi, early stop)', 'DAC', 'GAIL', 'BC', 'BC (early stop)'],
]
algo_cmap_i = [
    [1, 4, 2, 5, 3, 6]
]

# writing parameters assuming option 3 above
fig_shape, plot_size, num_stds, font_size, eval_interval, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
    include_expert_baseline = plot_common.get_fig_defaults(num_plots=len(num_datas))

half_width = True
if half_width:
    plot_size = [2, 2.4]
    algo_titles = [
        ['BC (multi)', 'BC (multi,\n early stop)', 'DAC', 'GAIL', 'BC', 'BC (early\n stop)'],
    ]
#####################################################################################################################

# pretty plotting, allow tex
# plt.rcParams.update({"font.family": "serif", "font.serif": "Times", "text.usetex": True, "pgf.rcfonts": False})
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

s_fig, s_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                            figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])
r_fig, r_axes = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1],
                            figsize=[plot_size[0] * fig_shape[1], plot_size[1] * fig_shape[0]])
if fig_shape[1] > 1: 
    s_axes_flat = s_axes.flatten()
else:
    s_axes_flat = [s_axes]
if fig_shape[1] > 1: 
    r_axes_flat = r_axes.flatten()
else:
    r_axes_flat = [r_axes]

# get returns and successes
all_returns = dict.fromkeys(task_dir_names)
all_successes = dict.fromkeys(task_dir_names)
for task_i, task in enumerate(task_dir_names):
    if not valid_task[task_i]:
        print(f"Task {task} set to false in valid_task, skipping")
        continue

    all_successes[task] = {}
    all_returns[task] = {}
    for num_data_i, num_data in enumerate(num_datas):
        all_successes[task][num_data] = {}
        all_returns[task][num_data] = {}
        for algo_i, algo in enumerate(algo_dir_names[num_data_i]):
            all_successes[task][num_data][algo] = dict(raw=[], mean=[], std=[])
            all_returns[task][num_data][algo] = dict(raw=[], mean=[], std=[])

    # all_successes[task] = { algo : {num_data: dict(raw=[], mean=[], std=[]) for num_data in num_datas} for algo in algo_dir_names }
    # all_returns[task] =  { algo : {num_data: dict(raw=[], mean=[], std=[]) for num_data in num_datas} for algo in algo_dir_names }
    for num_data_i, num_data in enumerate(num_datas):
        for algo_i, algo in enumerate(algo_dir_names[num_data_i]):
        
            # folder structure is task/seed/algo/experiment_name/datetime
            # experiment_root_dir, algo_dir_and_name = data_locations[task][num_data][algo]
            # algo_dir, experiment_name = algo_dir_and_name.split('/')

            algo_dir, experiment_name = data_locations[task][num_data][algo].split('/')

            data_path = os.path.join(experiment_root_dir, task, '1', algo_dir)
            if not os.path.exists(data_path):
                print("No path found at %s for task %s algo %s, moving on in data cleaning" % (data_path, task, algo))
                continue

            for seed in seeds:

                #==================================================
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
                if algo in multitask_algos or 'lfgp' in algo or 'multitask' in algo:
                    suc_fixed = []
                    ret_fixed = []

                    for aux_i in range(num_aux[task_i]):
                        rets_slice = slice(aux_i * eval_eps_per_task, aux_i * eval_eps_per_task + eval_eps_per_task)
                        # for all algos, -1 index is episode, -2 is which aux, 0 is eval step index
                        suc_fixed.append(suc_data[..., aux_i, rets_slice])
                        ret_fixed.append(ret_data[..., aux_i, rets_slice])

                    suc_data = np.array(suc_fixed)
                    ret_data = np.array(ret_fixed)                    
                    # now order is eval step index, aux index, eval ep index
                    if not 'valid' in algo:
                        suc_data = np.swapaxes(suc_data, 0, 1)
                        ret_data = np.swapaxes(ret_data, 0, 1)
                
                # remove extra eval step indices if there are any
                if not 'valid' in algo:
                    suc_data = suc_data[..., :num_eval_steps_to_use[task_i], :]
                    ret_data = ret_data[..., :num_eval_steps_to_use[task_i], :]

                try:
                    # all_successes[task][algo][num_data]['raw'].append(suc_data)
                    # all_returns[task][algo][num_data]['raw'].append(ret_data)
                    all_successes[task][num_data][algo]['raw'].append(suc_data)
                    all_returns[task][num_data][algo]['raw'].append(ret_data)
                except:
                    import ipdb; ipdb.set_trace()

                #==================================================

            # all_returns[task][algo][num_data]['raw'] = np.array(all_returns[task][algo][num_data]['raw']).squeeze()
            # all_successes[task][algo][num_data]['raw'] = np.array(all_successes[task][algo][num_data]['raw']).squeeze()
            all_returns[task][num_data][algo]['raw'] = np.array(all_returns[task][num_data][algo]['raw']).squeeze()
            all_successes[task][num_data][algo]['raw'] = np.array(all_successes[task][num_data][algo]['raw']).squeeze()

            # take along episode axis, then along seed axis
            for dat in [all_returns[task][num_data][algo], all_successes[task][num_data][algo]]:
                dat['mean'] = dat['raw'].mean(axis=-1).mean(axis=0)
                dat['std'] = dat['raw'].mean(axis=-1).std(axis=0)

# plotting
for task_i, task in enumerate(task_dir_names):
    for num_data_i, num_data in enumerate(num_datas):
        for algo_i, algo in enumerate(algo_dir_names[num_data_i]):
            ax_i = num_data_i
            s_ax = s_axes_flat[ax_i]
            r_ax = r_axes_flat[ax_i]
            # if algo not in all_successes[task].keys():
            #     print(f"No data for algo {algo} and task {task}, skipping")
            #     continue
            for ax, task_algo_amt_data in zip([s_ax, r_ax], [all_successes[task][num_data][algo], all_returns[task][num_data][algo]]):
                if algo in multitask_algos or 'lfgp' in algo or 'multitask' in algo:
                    mean = task_algo_amt_data['mean'][..., main_task_i[task_i]]
                    std = task_algo_amt_data['std'][..., main_task_i[task_i]]
                    raw = task_algo_amt_data['raw'][..., main_task_i[task_i]]
                else:
                    mean = task_algo_amt_data['mean']
                    std = task_algo_amt_data['std']
                    raw = task_algo_amt_data['raw']

                if algo in multitask_algos or 'lfgp' in algo or 'multitask' in algo:
                    line_style = '-'
                else:
                    line_style = '--'
                
                if 'valid' in algo:  # horizontal line
                    ax.axhline(mean, label=algo_titles[num_data_i][algo_i] if task_i == 0 else "",
                               color=cmap(algo_cmap_i[num_data_i][algo_i]), linewidth=linewidth, linestyle=line_style)
                    ax.axhspan(mean - num_stds * std, mean + num_stds * std, color=cmap(algo_cmap_i[num_data_i][algo_i]), 
                               alpha=std_alpha / 2)

                else:
                    x_vals = np.array(range(eval_interval, eval_interval * len(mean) + 1, eval_interval * subsample_rate))
                    x_vals = x_vals / x_val_scale

                    # test combining data at subsample rate to see if it reduces noise
                    if subsample_rate > 1:
                        new_mean = []
                        for num_data in num_datas:
                            new_samp = task_algo_amt_data[num_data]['raw'][:, task_algo_amt_data['raw'].shape[1] - subsample_rate:, ...]
                            if algo in multitask_algos:
                                new_mean.append(new_samp.mean(axis=(1, -1))[:, main_task_i[task_i]].mean())
                            else:
                                new_mean.append(new_samp.mean(axis=(1, -1)).mean())
                        mean = np.array(new_mean)

                    try:

                        ax.plot(x_vals, mean, label=algo_titles[num_data_i][algo_i],
                            color=cmap(algo_cmap_i[num_data_i][algo_i]), linewidth=linewidth, linestyle=line_style)
                        ax.fill_between(x_vals, mean - num_stds * std, mean + num_stds * std, facecolor=cmap(algo_cmap_i[num_data_i][algo_i]),
                                        alpha=std_alpha)
                    except:
                        import ipdb; ipdb.set_trace()

            # pretty up plot, add expert baselines
            for plot_type, ax in zip(['s', 'r'], [s_ax, r_ax]):
                # baselines
                if include_expert_baseline:
                    exp_data = pickle.load(open(expert_perf_files[task_i], 'rb'))
                    if plot_type == 's':
                        perf_mean = exp_data['executed_task_successes'][expert_perf_file_main_task_i[task_i]].mean()

                    else:
                        perf_mean = exp_data['executed_task_returns'][expert_perf_file_main_task_i[task_i]].mean()

                    
                    ax.axhline(perf_mean, label="Expert" if algo_i == 5 and num_data_i == 0 else "", color="k",
                            linewidth=linewidth, linestyle='-.')

                # pretty
                ax.grid(alpha=0.5)
                if half_width:
                    ax.set_title(num_datas_labels[num_data_i], fontsize=font_size, x=0.8)
                else:
                    ax.set_title(num_datas_labels[num_data_i], fontsize=font_size)
                if plot_type == 's':
                    ax.set_ylim(-.01, 1.01)

# plt.tight_layout()  # careful with this...only applies to one fig and doesn't seem to work right
for fig, fig_name in zip([s_fig, r_fig], ['s_fig.pdf', 'r_fig.pdf']):

    if half_width:
        fig_name = "halfw_" + fig_name

    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # ax.set_xlabel("Environment Steps (millions)", fontsize=font_size-2)
    if half_width:
        ax.set_xlabel("Updates/steps (millions)", fontsize=font_size, x=0.8)
    else:
        ax.set_xlabel("Updates/steps (millions)", fontsize=font_size)
    # ax.xaxis.set_label_coords(.57, -.15)  # if we have the 10^6 scientific notation
    if 's_fig' in fig_name:
        ax.set_ylabel("Success Rate", fontsize=font_size)
    else:
        ax.set_ylabel("Episode Return", fontsize=font_size)

    if half_width:
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-6, loc="center right", ncol=1, bbox_to_anchor=(1.575, 0.5))
    else:
        fig.legend(fancybox=True, shadow=True, fontsize=font_size-10, loc="lower center",
                ncol=int(math.ceil((len(algo_dir_names) + 1))), 
                # bbox_to_anchor=(0.5, -0.3))  
                bbox_to_anchor=(0.45, -0.375))

    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')