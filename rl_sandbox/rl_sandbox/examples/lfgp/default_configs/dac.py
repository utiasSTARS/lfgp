import argparse
import numpy as np
import os
import torch

import rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state as p_aux
import rl_sandbox.constants as c
import rl_sandbox.examples.lfgp.experiment_utils as exp_utils


from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import FullyConnectedSeparate, \
    FullyConnectedSquashedGaussianSAC
import rl_sandbox.examples.lfgp.default_configs.common as common_default

def reward_func(reward, **kwargs): return np.array([reward])

def get_settings(args):
    common_default.main_task_alias_set(args)
    obs_dim, action_dim = common_default.get_obs_action_dim(args)
    common_default.default_settings(args)
    device = torch.device(args.device)
    num_tasks = 1

    if args.env_type == c.MANIPULATOR_LEARNING:
        save_path = exp_utils.get_save_path(c.DAC, args.main_task, args.seed, args.exp_name, args.top_save_path)
    else:
        save_path = exp_utils.get_save_path(c.DAC, args.env_name, args.seed, args.exp_name, args.top_save_path)

    # expert path
    expert_buffer = os.path.join(args.expert_top_dir, args.expert_dir_rest, args.expert_filenames)

    # reward options -- ensure we get the correct aux reward
    if args.env_type == c.MANIPULATOR_LEARNING:
        aux_reward_all = p_aux.PandaPlayXYZStateAuxiliaryReward(args.main_task, include_main=False)
        aux_reward_names = [func.__qualname__ for func in aux_reward_all._aux_rewards]

        if "unstack" in args.main_task:
            aux_reward_name = "stack_0"
        elif "insert" in args.main_task:
            aux_reward_name = "insert_0"
        else:
            aux_reward_name = args.main_task

        if 'no_move' in aux_reward_name:
            task_name = aux_reward_name.split('_no_move_')[0]
            aux_reward_name = f"{task_name}_0"

        eval_reward = aux_reward_all._aux_rewards[aux_reward_names.index(aux_reward_name)]
        # eval_reward = reward_func  # use this for env reward

    elif args.env_type in [c.SAWYER, c.HAND_DAPG]:
        eval_reward = reward_func

    else:
        raise NotImplementedError("Not yet implemented for other env types")

    buffer_settings, expert_buffer_settings = common_default.get_buffer_settings(
        args, obs_dim, action_dim, num_tasks, False, device)

    ##### populate settings dictionary #####
    experiment_setting = {
        **common_default.get_rl_settings(args, obs_dim, action_dim, args.num_evals_per_task),
        **common_default.get_train_settings(args, action_dim, device),
        c.DISCRIMINATOR_SETTING: common_default.get_discriminator_settings(args, obs_dim, action_dim, num_tasks, device),
        c.OPTIMIZER_SETTING: common_default.get_optimizer_settings(args),
        c.BUFFER_SETTING: buffer_settings,
        c.EXPERT_BUFFER_SETTING: expert_buffer_settings,

        # Model
        c.MODEL_SETTING: {
            c.MODEL_ARCHITECTURE: FullyConnectedSeparate if args.no_shared_layers else FullyConnectedSquashedGaussianSAC,
            c.KWARGS: {
                **common_default.get_model_kwargs(args, obs_dim, action_dim, device),
            }
        },

        # DAC
        c.EXPERT_BUFFER: expert_buffer,
        c.EXPERT_AMOUNT: int(args.expert_amounts),
        c.EVALUATION_REWARD_FUNC: eval_reward,

        # Save
        c.SAVE_PATH: save_path,
    }

    if args.full_traj_expert_filenames:
        experiment_setting[c.FT_EXPERT_BUFFER] = os.path.join(args.expert_top_dir, args.ft_expert_dir_rest, args.expert_filenames)

    exp_utils.config_check(experiment_setting, args.top_save_path)

    return experiment_setting

