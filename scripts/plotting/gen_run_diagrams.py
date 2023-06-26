import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import time
import pickle

from manipulator_learning.sim.envs import *
import manipulator_learning.learning.data.img_depth_dataset as img_depth_dataset

from ast import literal_eval
from functools import partial

import rl_sandbox.constants as c

from rl_sandbox.examples.eval_tools.utils import load_model
from rl_sandbox.learning_utils import evaluate_policy
from rl_sandbox.utils import set_seed

def default_reward(reward, **kwargs):
    return np.array([reward])

# CONFIG
#--------------------------------------------------------------------------------
require_suc = False
num_episodes = 3
# seed = 0
seed = 1
change_seed_between_models = True
# device = 0
device = 'cpu'
starting_ep = 0
# starting_ep = 27  # for stack
save = True

# env options
render = True
# max_ep_timestep = 180
max_ep_timestep = 360
# max_ep_timestep = 90
cam_str = 'panda_play_alt_blue_side_cam'
end_on_success = False

# policy options
# task_str = 'stack'
task_strs = ['stack']
# method_str = 'dac'
method_strs = ['lfgp_wrs']
config_name = "sacx_experiment_setting.pkl"
deterministic = False

# intention options
forced_schedule = None
# forced_schedule = {0: 3, 20: 1}  # for getting reach+close data
# forced_schedule = {0: 3, 20: 0}  # for getting reach+open data
forced_schedule_order_only = None

top = "/media/starslab/users/trevor-ablett/dac-x/play_xyz"

fig_type = "final_perf"
# fig_type = "comparison"

if fig_type == "comparison":
    forced_schedule_order_only = None
    model_names = ["199999.pt", "399999.pt", "599999.pt", "799999.pt", "999999.pt"]
    task_strs = ['stack']
    num_episodes = 3
    end_on_success = False
    change_seed_between_models = True
    deterministic = False
    max_ep_timestep = 360

elif fig_type == "final_perf":
    forced_schedule_order_only = [2, 2, 2, 2, 2, 2, 2]
    method_strs = ["lfgp_wrs", "dac", "multitask_bc", "bc"] 
    # task_strs = ["stack", "unstack-stack", "bring", "insert"]
    task_strs = ["bring", "insert"]
    # task_strs = ["unstack-stack"]
    num_episodes = 5
    end_on_success = True
    change_seed_between_models = False
    deterministic = True
    max_ep_timestep = 90

# skips = []
skips = [['stack', 'lfgp_wrs'],
         ['stack', 'dac'],
         ['stack', 'multitask_bc'],
         ['stack', 'bc']]


# model_name = "399999.pt"

# model_names = ["599999.pt", "799999.pt", "999999.pt"]
# model_names = ["199999.pt"]

env = None

for task_str in task_strs:
    for method_str in method_strs:

        if [task_str, method_str] in skips:
            continue

        # for final performance models
        if fig_type == 'final_perf':
            if task_str == 'insert':
                if 'bc' in method_str:
                    model_names = ['39.pt']
                else:
                    model_names = ["3999999.pt"]
            else:
                if 'bc' in method_str:
                    model_names = ['19.pt']
                else:
                    model_names = ["1999999.pt"]

        # lfgp
        if method_str == 'lfgp_wrs':
            if task_str == "stack":
                if fig_type == 'comparison':
                    model_dir = top + "/stack_0/5/lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_best/09-14-22_11_50_05"
                elif fig_type == 'final_perf':
                    model_dir = top + "/stack_0/3/lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_best/09-14-22_11_50_06"
            elif task_str == "unstack-stack":
                model_dir = top + "/unstack_stack_env_only_0/1/lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-best/09-19-22_12_25_29"
            elif task_str == 'bring':
                model_dir = top + "/bring_0/1/lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-best/09-19-22_12_34_43"
            elif task_str == 'insert':
                model_dir = top + "/insert_0/3/lfgp_wrs/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-4M/09-21-22_16_45_39"
            else:
                raise NotImplementedError()

            # forced_schedule_order_only = [4, 3, 5, 2, 0, 4, 3, 5]

        elif method_str == 'dac':
            if task_str == "stack":
                model_dir = top + "/stack_0/1/dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2/09-19-22_22_55_42"
            elif task_str == "unstack-stack":
                model_dir = top + "/unstack_stack_env_only_0/1/dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2/09-20-22_18_48_24"
            elif task_str == 'bring':
                model_dir = top + "/bring_0/1/dac/4800_steps-90_sp_point5_play_open_1200_extra_lastsralv2/09-21-22_12_22_44"
            elif task_str == 'insert':
                model_dir = top + "/insert_0/1/dac/5600_steps-90_sp_point5_play_open_1400_extra_lasts_ralv2-4M/09-29-22_13_40_25"
            else:
                raise NotImplementedError()

        elif method_str == 'multitask_bc':
            if task_str == "stack":
                model_dir = top + "/stack_0/1/multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2/09-21-22_15_03_16"
            elif task_str == "unstack-stack":
                model_dir = top + "/unstack_stack_env_only_0/1/multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2/09-21-22_15_47_59"
            elif task_str == 'bring':
                model_dir = top + "/bring_0/1/multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2/09-21-22_15_10_19"
            elif task_str == 'insert':
                model_dir = top + "/insert_0/1/multitask_bc/800_steps-90_sp_point5_play_open_200_extra_lasts_ralv2-4M/09-29-22_13_40_19"
            else:
                raise NotImplementedError()

        elif method_str == 'bc':
            if task_str == "stack":
                model_dir = top + "/stack_0/1/bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M/09-20-22_11_03_19"
            elif task_str == "unstack-stack":
                model_dir = top + "/unstack_stack_env_only_0/1/bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M/09-20-22_11_06_19"
            elif task_str == 'bring':
                model_dir = top + "/bring_0/1/bc/4800_steps-90_sp_point5_play_open_1200_extra_lasts_ralv2-2M/09-20-22_11_05_12"
            elif task_str == 'insert':
                model_dir = top + "/insert_0/1/bc/5600_steps-90_sp_point5_play_open_1400_extra_lasts_ralv2-4M/09-22-22_10_37_08"
            else:
                raise NotImplementedError()




        # dac-x
        # model_dir = "/home/trevor/remote/monolith/dac-x/results/dacx_paper_experiments/panda_play/stack_0/3/dac-x/reset-9000_steps-90_sp_play_open-q-0.6_polyak/07-25-21_09_04_25"
        # model_dir = top + "/insert_0/1/dac-x/reset-9000_steps-90_sp_play_open-q-0.6_polyak/08-13-21_09_33_12"  # 58%
        # model_dir = top + "/unstack_stack_env_only_0/4/dac-x/35M_all_9000_steps/08-31-21_16_11_21"  # 76%
        # model_dir = top + "/stack_0/2/dac-x/reset-9000_steps-90_sp_play_open-cond-0.6_polyak/08-02-21_11_02_41/"
        # model_dir = top + "/stack_0/2/dac-x/reset-9000_steps-90_sp_play_open-q-0.6_polyak/07-25-21_09_04_23/"  # this one for analysis

        # bc
        # model_dir = "/home/trevor/remote/monolith/dac-x/results/dacx_paper_experiments/panda_play/stack_0/3/bc/reset-54000_steps-90_sp_suc_only-100_overfit_tolerance/06-24-21_23_14_57"
        # model_dir = top + "/unstack_stack_env_only_0/1/bc/35M_all_9000_steps/09-03-21_14_52_53"  # 82%

        # bc low data
        # model_dir = "/home/trevor/remote/monolith/dac-x/results/dacx_paper_experiments/panda_play/stack_0/3/bc_low_data/reset_9000_steps-90_sp/06-18-21_11_52_00"
        # model_dir = top + "/insert_0/5/bc_low_data/reset-9000_steps-90_sp/06-23-21_20_17_06"  # 6%
        # model_dir = top + "/unstack_stack_env_only_0/1/bc/35M_all_9000_steps-less-data/09-03-21_14_54_43"  # 14%

        # dac
        # model_dir = "/home/trevor/remote/monolith/dac-x/results/dacx_paper_experiments/panda_play/stack_0/3/dac/reset-54000_steps-90_sp_suc_only/06-21-21_18_59_25"
        # model_dir = top + "/insert_0/5/dac/reset-63000_steps-90_sp/06-18-21_07_52_54"  # 26%
        # model_dir = top + "/unstack_stack_env_only_0/1/dac/reset-54000_steps-90_sp_play_open/09-02-21_11_04_22"  # 0%
        # model_dir = top + "/stack_0/2/dac/reset-54000_steps-90_sp_suc_only/06-21-21_18_54_52"  # this one for analysis

        # multi-bc
        # model_dir = "/home/trevor/remote/monolith/dac-x/results/dacx_paper_experiments/panda_play/stack_0/3/multitask_bc/9000_steps-90_sp_play_open/08-04-21_13_02_24"
        # model_dir = top + "/insert_0/2/multitask_bc/9000_steps-90_sp_play_open/08-16-21_12_18_05"  # 22%
        # model_dir = top + "/unstack_stack_env_only_0/2/multitask_bc/35M_only_stack_unstack_all_9000_steps-90_sp_play_open/08-31-21_12_04_08"  # 54%

        # multi-dac
        # model_dir = "/home/trevor/remote/monolith/dac-x/results/dacx_paper_experiments/panda_play/stack_0/5/multitask_dac/reset-9000_steps-90_sp_play_open/08-17-21_07_30_13"
        # model_dir = top + "/insert_0/5/multitask_dac/reset-9000_steps-90_sp_play_open/08-22-21_13_41_27"  # 14%
        # model_dir = top + "/unstack_stack_env_only_0/4/multitask_dac/35M_all_9000_steps/09-04-21_08_31_12"  # 0%

        # task_str = model_dir.split('/')[7]
        # method_str = model_dir.split('/')[9]

        if method_str in ['bc', 'bc_low_data', 'dac']:
            eval_intention = 0
        else:
            eval_intention = 2

        if method_str == 'lfgp_wrs':
            config_name = "lfgp_experiment_setting.pkl"
        elif method_str == 'dac':
            config_name = "dac_experiment_setting.pkl"
        elif method_str == 'multitask_bc':
            config_name = "multitask_bc_experiment_setting.pkl"
        elif method_str == 'bc':
            config_name = "bc_experiment_setting.pkl"

        # for unstack stack ex vid DACX
        # forced_schedule_order_only = [2, 4, 4, 2, 5, 3, 4, 3]  # unstack-stack, 200k
        # forced_schedule_order_only = [2, 4, 0, 4, 0, 3, 5, 1]  # unstack-stack, 200k TWO
        # forced_schedule_order_only = [2, 2, 3, 2, 1, 0, 1, 5]  # unstack-stack, 400k
        # forced_schedule_order_only = [2, 2, 2, 3, 5, 5, 1, 3]  # unstack-stack, 400k TWO
        # forced_schedule_order_only = [2, 2, 2, 2, 4, 2, 1, 1]  # unstack-stack, 600k
        # forced_schedule_order_only = [2, 2, 2, 2, 5, 0, 3, 5]  # unstack-stack, 800k
        # forced_schedule_order_only = [2, 2, 2, 2, 2, 4, 5, 2]  # unstack-stack, 1M
        # forced_schedule_order_only = [2, 2, 2, 2, 2, 2, 2, 2]  # unstack-stack, 1.2M

        # for stack ex vid DACX COND SCHED
        # forced_schedule_order_only = [2, 3, 2, 0, 2, 2, 5, 2]  # 200k
        # forced_schedule_order_only = [2, 4, 4, 2, 0, 5, 3, 2]  # 1M
        # forced_schedule_order_only = [2, 0, 2, 2, 4, 4, 5, 2]  # 2M
        # forced_schedule_order_only = [3, 3, 5, 2, 2, 2, 2, 0]  # 3M

        # for stack ex vid DACX Q SCHED
        # forced_schedule_order_only = [2, 4, 4, 3, 5, 0, 3, 2]  # 200k
        # forced_schedule_order_only = [2, 2, 3, 2, 2, 5, 2, 2]  # 1M
        # forced_schedule_order_only = [2, 0, 2, 2, 3, 0, 1, 1]  # 2M
        # forced_schedule_order_only = [2, 2, 2, 2, 0, 2, 0, 2]  # 3M

        # uniform_scheduler_period = None
        # uniform_scheduler_suc = True
        uniform_scheduler_suc = False


        for model_name in model_names:
            all_ep_schedules = []

            main_exp_dir = '/media/trevor/Data/paper-data/dac-x/figures/runs_ralv2'
            # main_exp_dir = '/media/starslab/users/trevor-ablett/dac-x/figures/runs/ral-v2'
            run_dir = main_exp_dir + '/runs/' + method_str + '/' + task_str + '_' + model_name.split('.')[0] + '_' \
                + str(num_episodes) + '_eps_' + datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            # debug_camera_angle = [[.36, -.19, .62], 1.2, -38.6, -548.4]  # side view
            debug_camera_angle = [[.24, -.13, .26], 1.6, -65.4, -704]  # top view from opposite of cam
            # ex_base_angles = np.linspace(-3 * np.pi / 16, np.pi / 16, num=4)

            #--------------------------------------------------------------------------------

            config_path = os.path.join(model_dir, config_name)
            model_path = os.path.join(model_dir, model_name)

            obs_shape = 60  # +1 for absorbing state
            act_shape = 4

            ds = img_depth_dataset.Dataset(run_dir + '/' + cam_str, state_dim=obs_shape, act_dim=act_shape)

            
            print(f"model: {model_path},\n config: {config_path}")
            if env is None:
                config, env, buffer_preprocess, agent = load_model(seed, config_path, model_path, eval_intention,
                                                                    device, include_disc=False, force_egl=True)
            else:
                config, buffer_preprocess, agent = load_model(seed, config_path, model_path, eval_intention,
                                                                    device, include_disc=False, force_egl=True, include_env=False)
            
            if change_seed_between_models: seed += 1
            set_seed(seed)

            env.seed(seed)

            pbc = env.env._pb_client

            # setting debug camera
            pbc.resetDebugVisualizerCamera(cameraTargetPosition=debug_camera_angle[0],
                            cameraDistance=debug_camera_angle[1], cameraPitch=debug_camera_angle[2],
                            cameraYaw=debug_camera_angle[3])

            # load up aux rewards and aux successes
            if c.AUXILIARY_REWARDS in config:
                auxiliary_reward = config[c.AUXILIARY_REWARDS].reward
                if hasattr(config[c.AUXILIARY_REWARDS], 'set_aux_rewards_str'):
                    config[c.AUXILIARY_REWARDS].set_aux_rewards_str()
            else:
                auxiliary_reward = lambda reward, **kwargs: np.array([reward])

            if hasattr(env, 'get_task_successes') and c.AUXILIARY_REWARDS in config and \
                    hasattr(config[c.AUXILIARY_REWARDS], '_aux_rewards_str'):
                auxiliary_success = partial(env.get_task_successes, tasks=config[c.AUXILIARY_REWARDS]._aux_rewards_str)
            elif hasattr(env, 'VALID_AUX_TASKS') and auxiliary_reward.__qualname__ in env.VALID_AUX_TASKS:
                auxiliary_success = partial(env.get_task_successes, tasks=[auxiliary_reward.__qualname__])
            else:
                auxiliary_success = None

            img_traj_data = []
            depth_traj_data = []
            ep_count = 0
            ep_i = 0
            eval_returns = []
            done_successes = []
            aux_successes = []
            while ep_count < num_episodes:
                ep_schedule = []
                if ep_i < starting_ep:
                    curr_obs = env.reset()
                    ep_i += 1
                    continue

                curr_obs = env.reset()
                eval_returns.append(0)
                buffer_preprocess.reset()
                curr_obs = buffer_preprocess(curr_obs)
                h_state = agent.reset()
                done = False
                done_successes.append(0)
                aux_successes.append([0])
                ts = 0
                ts_since_suc = 0

                if uniform_scheduler_suc:
                    new_int = np.random.randint(agent.high_level_model._num_tasks)
                    if task_str == 'unstack_stack_env_only_0':
                        new_int = np.random.randint(agent.high_level_model._num_tasks - 1)
                        if new_int >= 3:
                            new_int += 1
                    print(f"at ts {ts}, switching to intention {new_int}")
                    agent.high_level_model._intention_i = new_int
                    agent.curr_high_level_act = new_int

                img, depth = env.env.render(cam_str)
                hq_traj_imgs = [img]
                hq_traj_depths = [np.array([0])]  # makes folders smaller

                traj_states = [np.concatenate([curr_obs.flatten(), np.zeros(act_shape).flatten(), np.array([0]), np.array([0]),
                                            np.array([False])])]
                while not done and ts < max_ep_timestep:
                    if hasattr(env, c.RENDER) and render:
                        env.render()

                    if 'lfgp' in method_str:
                        if forced_schedule is not None:
                            for t_key in forced_schedule.keys():
                                if ts == t_key:
                                    print(f"switching to intention {forced_schedule[ts]}")
                                    agent.high_level_model._intention_i = forced_schedule[ts]
                                    agent.curr_high_level_act = forced_schedule[ts]
                        if forced_schedule_order_only is not None:
                            if ts % 45 == 0:
                                new_int = ts // 45
                                print(f"switching to intention {forced_schedule_order_only[new_int]}")
                                agent.high_level_model._intention_i = np.array(forced_schedule_order_only[new_int])
                                agent.curr_high_level_act = np.array(forced_schedule_order_only[new_int])

                    if deterministic:
                        action, h_state, act_info = agent.deterministic_action(obs=curr_obs, hidden_state=h_state)
                    else:
                        action, h_state, act_info = agent.compute_action(obs=curr_obs, hidden_state=h_state)
                    
                    if ts % 45 == 0 and method_str == 'lfgp_wrs':
                        ep_schedule.append(int(agent.curr_high_level_act))

                    if config[c.CLIP_ACTION]:
                        action = np.clip(action, a_min=config[c.MIN_ACTION], a_max=config[c.MAX_ACTION])

                    next_obs, reward, done, env_info = env.step(action)
                    next_obs = buffer_preprocess(next_obs)

                    if env_info.get(c.DONE_SUCCESS, False) or (env_info.get(c.INFOS, [{}])[0].get(c.DONE_SUCCESS, False)):
                        done_successes[-1] = 1

                    if auxiliary_success is not None:
                        aux_successes[-1] = np.array(auxiliary_success(observation=curr_obs, action=action,
                                                                    env_info=env_info['infos'][-1])).astype(int).tolist()

                    eval_returns[-1] += np.atleast_1d(auxiliary_reward(observation=curr_obs, action=action, reward=reward,
                                                                    done=done, next_observation=next_obs, info=env_info))
                    curr_obs = next_obs
                    ts += 1
                    print(f"ep {ep_count + 1} of {num_episodes}, ts {ts} of {max_ep_timestep}")

                    # if auxiliary_success is not None:
                    #     print(eval_returns[-1], aux_successes[-1])
                    # else:
                    #     print(eval_returns[-1])

                    img, depth = env.env.render(cam_str)
                    hq_traj_imgs.append(img)
                    hq_traj_depths.append(np.array([0]))
                    traj_states.append(np.concatenate([curr_obs.flatten(), np.array(action).flatten(), np.array([reward]),
                                                    np.array([0]), np.array([done])]))

                    if uniform_scheduler_suc:
                        ts_since_suc += 1
                        if aux_successes[-1][agent.high_level_model._intention_i] or ts_since_suc % 90 == 0:
                            if aux_successes[-1][agent.high_level_model._intention_i]:
                                print("Succeeded!!")
                            new_int = np.random.randint(agent.high_level_model._num_tasks)
                            if task_str == 'unstack_stack_env_only_0':
                                new_int = np.random.randint(agent.high_level_model._num_tasks - 1)
                                if new_int >= 3:
                                    new_int += 1
                            print(f"at ts {ts}, switching to intention {new_int}")
                            agent.high_level_model._intention_i = new_int
                            agent.curr_high_level_act = new_int
                            env.reset_episode_success_data()
                            ts_since_suc = 0



                    if end_on_success:
                        if auxiliary_success is not None:
                            if aux_successes[-1][eval_intention]:
                                break
                        else:
                            if done_successes[-1]:
                                break

                ep_i += 1
                all_ep_schedules.append(ep_schedule)
                if require_suc:
                    if aux_successes[-1][eval_intention]:
                        ep_count += 1
                        if save:
                            ds.append_traj_data_lists(traj_states, hq_traj_imgs, hq_traj_depths, final_obs_included=True)
                else:
                    ep_count += 1
                    if save:
                        ds.append_traj_data_lists(traj_states, hq_traj_imgs, hq_traj_depths, final_obs_included=True)
                
                if method_str == 'lfgp_wrs':
                    with open(run_dir + '/scheds.txt', 'a') as f:
                        f.write(f"{ep_schedule} \n")
