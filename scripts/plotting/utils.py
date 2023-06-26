import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import common as plot_common


MULTITASK_ALGOS = ['multitask_bc', 'dac-x', 'multitask_dac']

def get_default_opts():
    opts = {}

    # path
    opts['root_dir'] = "/media/raid5-array/experiments/dac-x"
    opts['fig_path'] = opts['root_dir'] + "/figures/main_performance"
    opts['experiment_root_dir'] = opts['root_dir'] + "/results/dacx_paper_experiments/panda_play"
    opts['seeds'] = ['1','2','3','4','5']
    opts['expert_root'] = os.path.join(opts['root_dir'], "play_xyz/expert-data")
    opts['expert_perf_files'] = [
        os.path.join(opts['expert_root'], "open-close-lift-reach-move/policies/05-01-21_15_17_07/eval_1499999_100_eps_per_int.pkl"),
        os.path.join(opts['expert_root'], "open-close-stack-lift-reach-move/policies/05-09-21_21_57_07/eval_1999999_100_eps_per_int.pkl"),
        os.path.join(opts['expert_root'], "open-close-insert-bring-lift-reach-move/policies/05-30-21_20_38_48/eval_1299999_100_eps_per_int.pkl"),
        os.path.join(opts['expert_root'], "open-close-insert-bring-lift-reach-move/policies/05-30-21_20_38_48/eval_2699999_100_eps_per_int.pkl")]
    opts['expert_perf_file_main_task_i'] = [2, 2, 3, 2]

    # task
    opts['task_dir_names'] = ["stack_0", "unstack_stack_env_only_0", "bring_0", "insert_0"]
    opts['valid_task'] = [True, True, True, True]
    opts['task_titles'] = ["Stack", "Unstack-Stack", "Bring", "Insert"]
    opts['main_task_i'] = [2, 2, 2, 2]
    opts['num_aux'] = [6, 6, 6, 7]
    opts['task_data_filenames'] = ['train.pkl', 'train.pkl', 'train_rerun.pkl', 'train.pkl']
    # opts['task_data_filenames'] = ['train.pkl', 'train.pkl', 'train.pkl', 'train.pkl']

    # algorithm
    # opts['algo_dir_names = ['dac-x', 'multitask_dac', 'multitask_bc', 'dac', 'bc']
    opts['algo_dir_names'] = ['dac-x', 'multitask_dac', 'multitask_bc', 'dac', 'bc', 'bc_low_data']  # NOTE: bc_low_data is actually in bc now
    # opts['algo_titles'] = ['LfGP (multi)', 'DAC (multi)', 'BC (multi)', 'DAC', 'BC']
    opts['algo_titles'] = ['LfGP (multi)', 'DAC (multi)', 'BC (multi)', 'DAC', 'BC', 'BC (less data)']
    # opts['algo_titles'] = [r'LfGP (multitask $\mathcal{B}_\mathcal{T}$)', 'DAC (multitask)', 'BC (multitask)', 'DAC', 'BC']
    opts['rl_eval_eps_per_task'] = 50
    opts['bc_eval_eps_per_task'] = 50

    # fig
    opts['fig_shape'] = [1, 4]  # row x col
    opts['plot_size'] = [3.2, 2.4]
    opts['num_stds'] = 1
    opts['font_size'] = 16
    opts['eval_interval'] = 100000
    opts['cmap'] = plt.get_cmap("tab10")
    opts['linewidth'] = 1
    opts['std_alpha'] = .5
    opts['x_val_scale'] = 1e6
    opts['subsample_rate'] = 1  # 1 for no subsample

    return opts

# get returns and successes
def get_returns_successes(fig_name, data_locations):
    root_dir, fig_path, experiment_root_dir, seeds, expert_root, expert_perf_files, expert_perf_file_main_task_i = \
        plot_common.get_path_defaults(fig_name=fig_name)

    task_dir_names, valid_task, task_titles, main_task_i, num_aux, task_data_filenames, num_eval_steps_to_use = \
        plot_common.get_task_defaults()
    
    algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task = plot_common.get_algo_defaults()

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
    
    return all_successes, all_returns

def plot_mean_std(ax, aux_i, algo, algo_i, task_algo_data, algo_label=None, **kwargs):
    algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task = plot_common.get_algo_defaults()
    fig_shape, plot_size, num_stds, font_size, eval_interval, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
        include_expert_baseline = plot_common.get_fig_defaults()
    
    if algo in multitask_algos:
        mean = task_algo_data['mean'][..., [aux_i]].squeeze()
        std = task_algo_data['std'][..., [aux_i]].squeeze()
    else:
        mean = task_algo_data['mean'].squeeze()
        std = task_algo_data['std'].squeeze()

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
                new_mean.append(new_samp.mean(axis=(1, -1))[:, [aux_i]].mean())
                new_std.append(new_samp.mean(axis=(1, -1))[:, [aux_i]].std())
            else:
                new_mean.append(new_samp.mean(axis=(1, -1)).mean())
                new_std.append(new_samp.mean(axis=(1, -1)).std())
        mean = np.array(new_mean)
        std = np.array(new_std)

    ax.plot(x_vals, mean, label=algo_label if not None else "",
            color=cmap(algo_i), linewidth=linewidth, linestyle=line_style)

    try:
        ax.fill_between(x_vals, mean - num_stds * std, mean + num_stds * std, facecolor=cmap(algo_i),
                        alpha=std_alpha)
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()