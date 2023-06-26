""" 
Common things for many plotting files. 
task_inds are: 0--stack, 1--unstack-stack, 2--bring, 3--insert
"""

import os
import matplotlib.pyplot as plt


ALGO_TITLE_DICT = {
    'lfgp_wrs': 'LfGP (multi)',
    'multitask_bc': 'BC (multi)',
    'dac': 'DAC (single)',
    'bc': 'BC (single)'
}

def get_path_defaults(fig_name, task_inds=(0,1,2,3)):
    root_dir = "/media/starslab/users/trevor-ablett/dac-x"
    fig_path = root_dir + "/figures/" + fig_name
    experiment_root_dir = root_dir + "/play_xyz"
    seeds = ['1','2','3','4','5']
    expert_root = os.path.join(root_dir, "play_xyz/expert-data")
    expert_perf_files = [
        os.path.join(expert_root, "open-close-stack-lift-reach-move/policies/05-09-21_21_57_07/eval_1999999_100_eps_per_int.pkl"),
        os.path.join(expert_root, "open-close-unstackstack-lift-reach-move-35M/policies/08-29-21_23_05_03/eval_3499999_100_eps_per_int.pkl"),
        os.path.join(expert_root, "open-close-insert-bring-lift-reach-move/policies/05-30-21_20_38_48/eval_1299999_100_eps_per_int.pkl"),
        os.path.join(expert_root, "open-close-insert-bring-lift-reach-move/policies/05-30-21_20_38_48/eval_2699999_100_eps_per_int.pkl")]
    expert_perf_file_main_task_i = [2, 2, 3, 2]

    out_epf = []
    out_epf_mti = []
    for i in task_inds:
        out_epf.append(expert_perf_files[i])
        out_epf_mti.append(expert_perf_file_main_task_i[i])

    return root_dir, fig_path, experiment_root_dir, seeds, expert_root, out_epf, out_epf_mti


def get_task_defaults(task_inds=(0,1,2,3)):
    task_dir_names = ["stack_0", "unstack_stack_env_only_0", "bring_0", "insert_0"]
    valid_task = [True, True, True, True]
    task_titles = ["Stack", "Unstack-Stack", "Bring", "Insert"]
    main_task_i = [2, 2, 2, 2]
    num_aux = [6, 6, 6, 7]
    task_data_filenames = ['train.pkl', 'train.pkl', 'train_rerun.pkl', 'train.pkl']
    num_eval_steps_to_use = [20, 20, 20, 40]

    out_tdn = []
    out_vt = []
    out_tt = []
    out_mti = []
    out_na = []
    out_tdf = []
    out_nestu = []
    for i in task_inds:
        out_tdn.append(task_dir_names[i])
        out_vt.append(valid_task[i])
        out_tt.append(task_titles[i])
        out_mti.append(main_task_i[i])
        out_na.append(num_aux[i])
        out_tdf.append(task_data_filenames[i])
        out_nestu.append(num_eval_steps_to_use[i])
    
    return out_tdn, out_vt, out_tt, out_mti, out_na, out_tdf, out_nestu


def get_algo_defaults():
    algo_dir_names=['lfgp_wrs', 'multitask_bc', 'dac', 'bc']
    algo_titles = ['LfGP (multi)', 'BC (multi)', 'DAC (single)', 'BC (single)']
    multitask_algos = ['multitask_bc', 'lfgp_wrs']
    eval_eps_per_task = 50

    return algo_dir_names, algo_titles, multitask_algos, eval_eps_per_task


def get_fig_defaults(num_plots=4):
    fig_shape = [1, num_plots]  # row x col
    plot_size = [3.2, 2.4]
    num_stds = 1
    font_size = 16
    eval_interval = 100000
    cmap = plt.get_cmap("tab10")
    linewidth = 1
    std_alpha = .5
    x_val_scale = 1e6
    subsample_rate = 1  # 1 for no subsample
    include_expert_baseline = True

    return fig_shape, plot_size, num_stds, font_size, eval_interval, cmap, linewidth, std_alpha, x_val_scale, subsample_rate, \
        include_expert_baseline