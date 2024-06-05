import argparse
import math
import numpy as np
import torch
import os

import rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state as p_aux
import rl_sandbox.auxiliary_rewards.rce_envs.sawyer as s_aux
import rl_sandbox.auxiliary_rewards.rce_envs.door_human_v0 as door_aux
import rl_sandbox.auxiliary_rewards.rce_envs.hammer_human_v0 as hammer_aux
import rl_sandbox.auxiliary_rewards.rce_envs.relocate_human_v0 as relocate_aux
import rl_sandbox.auxiliary_rewards.generic as generic_aux
import rl_sandbox.constants as c
import rl_sandbox.examples.lfgp.experiment_utils as exp_utils
import rl_sandbox.examples.lfgp.transfer as transfer
import rl_sandbox.transforms.general_transforms as gt

from rl_sandbox.agents.random_agents import UniformContinuousAgent
from rl_sandbox.algorithms.sac_x.schedulers import \
    WeightedRandomScheduler, RecycleScheduler, WeightedRandomSchedulerPlusHandcraft, QTableScheduler, FixedScheduler
from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.absorbing_state import AbsorbingStateWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import \
    MultiTaskFullyConnectedSquashedGaussianSAC, MultiTaskFullyConnectedSquashedGaussianTD3
from rl_sandbox.model_architectures.discriminators.fully_connected_discriminators import ActionConditionedFullyConnectedDiscriminator
from rl_sandbox.model_architectures.layers_definition import VALUE_BASED_LINEAR_LAYERS, SAC_DISCRIMINATOR_LINEAR_LAYERS
from rl_sandbox.train.train_lfgp_sac import train_lfgp_sac
import rl_sandbox.examples.lfgp.default_configs.common as common_default
import rl_sandbox.examples.lfgp.default_configs.common_args as common_args


def get_parser():
    parser = common_args.get_parser()

    # Multi/single
    parser.add_argument('--single_task', action="store_true", help="Run single task instead of multi.")

    # env, for multitask
    parser.add_argument('--main_intention', type=int, default=2, help="The main intention index")

    # data
    parser.add_argument('--load_existing_dir', type=str, default="", help="dir with existing model, buffer, and exp settings.")
    parser.add_argument('--load_model', type=str, default="", help="Path for model to be loaded")
    parser.add_argument('--load_buffer', type=str, default="", help="Path for buffer to be loaded")
    parser.add_argument('--load_transfer_exp_settings', type=str, default="",
                        help="The experimental settings of a previous run. If set, transfer any possible auxiliaries"
                            "from this old model to the new one.")
    parser.add_argument('--load_max_buffer_index', type=int, required=False, help="If transferring, max buffer index.")
    parser.add_argument('--load_aux_old_removal', type=str, required=False, default="",
                        help="comma sep list of aux tasks from old model to ignore for transfer.")

    # lfgp/discriminator
    parser.add_argument('--no_branched_outputs', action="store_true", help="Branched vs. shared outputs on q/policy.")
    parser.add_argument('--scheduler', type=str, choices=['wrs_plus_handcraft', 'wrs', 'learned', 'no_sched'],
                        default="wrs_plus_handcraft")
    parser.add_argument('--scheduler_period', type=int, default=45, help="SAC-X fixed scheduler period before policy switch.")
    parser.add_argument('--task_shared_layers_only', action='store_true', help="Overrides no_shared_layers. If set, "\
                        "each task has own set of shared layers, but policy/qs do not share layers.")
    parser.add_argument('--main_task_loss_weight', type=float, default=1.0,
                        help="Value for upweighting main task loss, and downweighting non main task losses correspondingly. "\
                             "e.g., setting this to 5.0 with 6 tasks would give main task 3.0, each other task 0.6")
    parser.add_argument('--shared_layers_width', type=int, default=256, help="Width of shared layers.")
    parser.add_argument('--num_extra_hidden', type=int, default=0, help="Number of extra hidden layers.")
    parser.add_argument('--no_disc_shared_layers', action="store_true",
                        help="No discriminator shared layers.")
    parser.add_argument('--disc_branched_outputs', action="store_true", help="Branched outputs on discriminator.")

    return parser

def get_settings(args):
    if args.env_type == c.MANIPULATOR_LEARNING:
        common_default.main_task_alias_set(args)
    obs_dim, action_dim = common_default.get_obs_action_dim(args)
    common_default.default_settings(args)
    device = torch.device(args.device)

    # get reward and num_tasks from env
    if args.env_type == c.MANIPULATOR_LEARNING:
        aux_reward = p_aux.PandaPlayXYZStateAuxiliaryReward(args.main_task, include_main=False)
    elif args.env_type == c.SAWYER:
        aux_reward = s_aux.SawyerAuxiliaryReward(args.env_name, aux_rewards=args.sawyer_aux_tasks.split(','))
    elif args.env_type == c.HAND_DAPG:
        if "door" in args.env_name:
            aux_reward = door_aux.DoorHumanV0AuxiliaryReward(args.env_name,
                                                             aux_rewards=args.hand_dapg_aux_tasks.split(','))
        elif "hammer" in args.env_name:
            aux_reward = hammer_aux.HammerHumanV0AuxiliaryReward(args.env_name,
                                                                 aux_rewards=args.hand_dapg_aux_tasks.split(','))
        elif "relocate" in args.env_name:
            aux_reward = relocate_aux.RelocateHumanV0AuxiliaryReward(args.env_name,
                                                                     aux_rewards=args.hand_dapg_aux_tasks.split(','))
        else:
            raise NotImplementedError(f"Aux reward not implemented for hand_dapg env {args.env_name}")
    elif args.env_type == c.PANDA_RL_ENVS:
        expert_filenames_list = args.expert_filenames.split(',')
        aux_reward = []
        for fn_i, fn_str in enumerate(expert_filenames_list):
            if fn_i == 0:
                aux_reward.append('main')
            else:
                fn_no_ext = fn_str.split('.gz')[0]
                # if fn_no_ext == args.env_name:
                #     aux_reward.append('main')
                # else:
                aux_reward.append(fn_no_ext.split('_')[-1])
    else:
        raise NotImplementedError("Not yet implemented for other env types!")

    if args.env_type in [c.MANIPULATOR_LEARNING, c.SAWYER, c.HAND_DAPG]:
        num_tasks = aux_reward.num_auxiliary_rewards
    else:
        num_tasks = len(aux_reward)

    # LfGP/SAC-X unchanging constants
    task_select_probs = [0.5 / (num_tasks - 1) for _ in range(num_tasks)]
    task_select_probs[args.main_intention] = 0.5
    num_evaluation_episodes = args.num_evals_per_task * num_tasks

    # get expert paths and amounts
    expert_paths_top = os.path.join(args.expert_top_dir, args.expert_dir_rest)
    filenames = [fn for fn in args.expert_filenames.split(',')]
    assert len(filenames) == num_tasks, f"Length of expert_filenames argument was {len(filenames)}, but num_tasks was {num_tasks}"
    expert_buffers = [os.path.join(expert_paths_top, filename) for filename in filenames]

    if args.expert_amounts == "":
        expert_amounts = [None] * num_tasks
    else:
        amounts = args.expert_amounts.split(',')
        if len(amounts) == 1:
            expert_amounts = [int(amounts[0])] * num_tasks
        else:
            expert_amounts = [int(am) for am in args.expert_amounts.split(',')]

    # for consistency with existing runs, we'll keep both wrs versions in lfgp_wrs
    exp_name_dir = "lfgp_" + str(args.scheduler)
    if args.scheduler == "wrs_plus_handcraft": exp_name_dir = "lfgp_wrs"

    if args.env_type == c.MANIPULATOR_LEARNING:
        save_path = exp_utils.get_save_path(exp_name_dir, args.main_task, args.seed, args.exp_name, args.top_save_path)
    else:
        save_path = exp_utils.get_save_path(exp_name_dir, args.env_name, args.seed, args.exp_name, args.top_save_path)

    # Loading existing model if set
    load_model, load_buffer, load_transfer_exp_settings, load_aux_old_removal = transfer.get_transfer_params(
        args.load_existing_dir, args.load_model, args.load_buffer, args.load_transfer_exp_settings, args.load_aux_old_removal)

    # set scheduler
    if args.scheduler == "wrs_plus_handcraft":
        if args.env_type == c.PANDA_RL_ENVS:
            if args.env_name in ['PandaDrawer', 'PandaDrawerLine']:
                ma = 0; re = 1; gr = 2
                handcraft_traj_epsilon = 1.0
                if args.scheduler_period == 20:
                    handcraft_traj_options = [
                        [re, gr, ma],
                        [re, gr, ma],
                        [re, gr, ma],
                        [ma, ma, ma],
                    ]
            elif args.env_name in [
                'PandaDoorAngle', 'PandaDoorAngleLongEp', 'PandaDoor', 
                'PandaDoorLongEp', 'PandaDrawerLongEp', 'PandaDrawerLineLongEp'
            ]:
                ma = 0; re = 1; gr = 2
                handcraft_traj_epsilon = 0.5
                if args.scheduler_period == 20:
                    handcraft_traj_options = [[ma, ma, ma]]
                    handcraft_traj_options[0].extend([re, gr, ma] * 15)
                    # handcraft_traj_options = [
                    #     [re, gr, ma] * 16
                    # ]
                    handcraft_traj_options[0].extend([re, gr])
            else:
                raise NotImplementedError("handcraft scheduler not yet set up for other panda rl envs!")
        elif args.env_type == c.MANIPULATOR_LEARNING:
            handcraft_traj_epsilon = .5
            if args.main_task == 'insert_0':
                handcraft_traj_options = [
                    [5, 4, 2, 0, 5, 4, 2, 0],
                    [5, 4, 6, 2, 0, 5, 4, 6],
                    [4, 2, 0, 4, 2, 0, 4, 2],
                    [2, 0, 2, 0, 2, 0, 2, 0],
                    [6, 2, 0, 6, 2, 0, 6, 2],
                    [3, 2, 0, 3, 2, 0, 3, 2],
                    [5, 4, 3, 2, 0, 5, 4, 3],
                ]
            elif args.main_task == 'move_obj_0':
                handcraft_traj_options = [
                    [3, 2, 4, 0, 3, 2, 4, 0],
                    [3, 1, 2, 4, 3, 1, 2, 4],
                    [4, 0, 4, 0, 4, 0, 4, 0],
                ]
            elif args.main_task == 'unstack_stack_0':  # the one that includes the unstack aux
                op = 0; cl = 1; st = 2; us = 3; li = 4; re = 5; mo = 6
                handcraft_traj_options = [
                    [us, re, li, st, op, re, li, st],
                    [us, re, li, mo, st, op, re, li],
                    [us, li, st, op, us, li, st, op],
                    [st, op, st, op, st, op, st, op],
                    [us, mo, st, op, mo, st, op, mo],
                    [us, re, li, op, us, re, li, op]
                ]
            elif 'no_move' in args.main_task:
                op = 0; cl = 1; st = 2; li = 3; re = 4
                handcraft_traj_options = [
                    [re, li, st, op, re, li, st, op],
                    [li, st, op, li, st, op, li, st],
                    [st, op, st, op, st, op, st, op],
                ]
            else:
                handcraft_traj_options = [
                    [4, 3, 2, 0, 4, 3, 2, 0],
                    [4, 3, 5, 2, 0, 4, 3, 5],
                    [3, 2, 0, 3, 2, 0, 3, 2],
                    [2, 0, 2, 0, 2, 0, 2, 0],
                    [5, 2, 0, 5, 2, 0, 5, 2],
                ]
        elif args.env_type == c.SAWYER:
            ma = 0; re = 1; gr = 2
            handcraft_traj_epsilon = 1.0
            if num_tasks == 2:
                if args.action_repeat == 3 and args.scheduler_period == 10:
                    handcraft_traj_options = [
                        [re, ma, ma, ma, ma],
                        [ma, ma, ma, ma, ma]
                    ]
                elif args.action_repeat == 2 and args.scheduler_period == 38:
                    handcraft_traj_options = [
                        [re, ma],
                        [ma, ma]
                    ]
                elif args.scheduler_period == 75:
                    handcraft_traj_options = [
                        [re, ma],
                        [ma, ma]
                    ]
                elif args.scheduler_period == 50:
                    handcraft_traj_options = [
                        [re, ma, ma],
                        [ma, ma, ma],
                    ]
                elif args.scheduler_period == 30:
                    handcraft_traj_options = [
                        [re, ma, ma, ma, ma],
                        [ma, ma, ma, ma, ma]
                    ]
                else:
                    raise NotImplementedError(f"wrs_plus_handcraft not implemented for sawyer + period other than 50, 75")
            elif num_tasks == 3:
                if args.scheduler_period == 30:
                    handcraft_traj_options = [
                        [re, gr, ma, ma, ma],
                        [ma, ma, ma, ma, ma],
                    ]
                else:
                    raise NotImplementedError(f"wrs_plus_handcraft not implemented for sawyer, 3 tasks, sched period {args.scheduler_period}")

        elif args.env_type == c.HAND_DAPG:
            ma = 0; re = 1; gr = 2
            handcraft_traj_epsilon = 1.0
            if num_tasks == 3:
                if args.scheduler_period in [40, 10, 7]:  # 40, 10 & 7 for frame_skip of 5 (default), 20 and 30 respectively
                    handcraft_traj_options = [
                        [re, gr, ma, ma, ma],
                        [ma, ma, ma, ma, ma],
                    ]
                elif args.scheduler_period == 25:
                    handcraft_traj_options = [
                        [re, gr, ma, ma, ma, ma, ma, ma],
                        [ma, ma, ma, ma, ma, ma, ma, ma],
                    ]
                else:
                    raise NotImplementedError(f"wrs_plus_handcraft not implemented for sawyer, 3 tasks, sched period {args.scheduler_period}")
            elif num_tasks == 2:
                if args.scheduler_period == 40:
                    handcraft_traj_options = [
                        [re, ma, ma, ma, ma],
                        [ma, ma, ma, ma, ma],
                    ]
        else:
            raise NotImplementedError(f"wrs_plus_handcraft not implemented for env type: {args.env_type}")

        train_scheduler_setting = {
            c.MODEL_ARCHITECTURE: WeightedRandomSchedulerPlusHandcraft,
            c.KWARGS: {
                c.TASK_SELECT_PROBS: task_select_probs,
                c.MAX_SCHEDULE: math.ceil(args.max_episode_length / args.scheduler_period),
                c.NUM_TASKS: num_tasks,
                "handcraft_traj_epsilon": handcraft_traj_epsilon,
                "handcraft_traj_options": handcraft_traj_options,
            },
            c.SCHEDULER_PERIOD: args.scheduler_period,
        }

    elif args.scheduler == "wrs":
        train_scheduler_setting = {
            c.MODEL_ARCHITECTURE: WeightedRandomScheduler,
            c.KWARGS: {
                c.TASK_SELECT_PROBS: task_select_probs,
                c.MAX_SCHEDULE: math.ceil(args.max_episode_length / args.scheduler_period),
                c.NUM_TASKS: num_tasks,
            },
            c.SCHEDULER_PERIOD: args.scheduler_period,
        }

    elif args.scheduler == "learned":
        train_scheduler_setting = {
            c.MODEL_ARCHITECTURE: QTableScheduler,
            c.KWARGS: {
                c.MAX_SCHEDULE: math.ceil(args.max_episode_length / args.scheduler_period),
                c.NUM_TASKS: num_tasks,
                c.TEMPERATURE: 360.,
                c.TEMPERATURE_DECAY: 0.9995,
                c.TEMPERATURE_MIN: 0.1,
            },
            c.SCHEDULER_PERIOD: args.scheduler_period,
        }

    elif args.scheduler == "no_sched":
        train_scheduler_setting = {
            c.MODEL_ARCHITECTURE: FixedScheduler,
            c.KWARGS: {
                c.INTENTION_I: args.main_intention,
                c.NUM_TASKS: num_tasks,
            },
            c.SCHEDULER_PERIOD: c.MAX_INT,
        }
    else:
        raise NotImplementedError(f"LfGP not implemented for scheduler selection {args.scheduler}")

    buffer_setting, expert_buffer_setting = common_default.get_buffer_settings(
        args, obs_dim, action_dim, num_tasks, load_buffer, device)

    ##### populate settings dictionary #####
    experiment_setting = {
        **common_default.get_rl_settings(args, obs_dim, action_dim, num_evaluation_episodes),
        **common_default.get_train_settings(args, action_dim, device),
        c.DISCRIMINATOR_SETTING: common_default.get_discriminator_settings(args, obs_dim, action_dim, num_tasks, device),
        c.OPTIMIZER_SETTING: common_default.get_optimizer_settings(args),
        c.BUFFER_SETTING: buffer_setting,
        c.EXPERT_BUFFER_SETTING: expert_buffer_setting,

        # Load
        c.LOAD_MODEL: load_model,
        c.LOAD_TRANSFER_EXP_SETTINGS: load_transfer_exp_settings,
        c.TRANSFER_PRETRAIN: 1000,  # TODO this should not be hardcoded!!!
        c.TRANSFER_BUFFER_DOWNSAMPLE: 1.0,
        c.TRANSFER_BUFFER_MAX_INDEX: args.load_max_buffer_index,
        c.TRANSFER_AUX_IGNORE: load_aux_old_removal,

        # Model
        c.INTENTIONS_SETTING: {
            c.MODEL_ARCHITECTURE: MultiTaskFullyConnectedSquashedGaussianSAC,
            # c.MODEL_ARCHITECTURE: MultiTaskFullyConnectedSquashedGaussianTD3,
            c.KWARGS: {
                **common_default.get_model_kwargs(args, obs_dim, action_dim, device),
                c.TASK_DIM: num_tasks,
                c.BRANCHED_OUTPUTS: not args.no_branched_outputs,
            },
        },

        # expert buffers
        c.EXPERT_BUFFERS: expert_buffers,
        c.EXPERT_AMOUNTS: expert_amounts,

        # SACX
        c.SCHEDULER_SETTING: {
            c.TRAIN: train_scheduler_setting,
            c.EVALUATION: {
                c.MODEL_ARCHITECTURE: RecycleScheduler,
                c.KWARGS: {
                    c.NUM_TASKS: num_tasks,
                    c.SCHEDULING: [num_evaluation_episodes // num_tasks] * num_tasks
                },
                c.SCHEDULER_PERIOD: c.MAX_INT,
            },
        },
        c.AUXILIARY_REWARDS: aux_reward,
        c.NUM_TASKS: num_tasks,
        c.SCHEDULER_TAU: 0.6,
        c.MAIN_INTENTION: args.main_intention,
        c.MAIN_TASK_LOSS_WEIGHT: args.main_task_loss_weight,

        # Save
        c.SAVE_PATH: save_path,
    }

    if args.full_traj_expert_filenames:
        ft_expert_paths_top = os.path.join(args.expert_top_dir, args.ft_expert_dir_rest)
        ft_filenames = [fn for fn in args.full_traj_expert_filenames.split(',')]
        assert len(ft_filenames) == num_tasks, f"Length of full_traj_expert_filenames argument was {len(ft_filenames)}, but num_tasks was {num_tasks}"
        ft_expert_buffers = [os.path.join(ft_expert_paths_top, filename) for filename in ft_filenames]
        experiment_setting[c.FT_EXPERT_BUFFERS] = ft_expert_buffers

    exp_utils.config_check(experiment_setting, args.top_save_path)

    return experiment_setting
