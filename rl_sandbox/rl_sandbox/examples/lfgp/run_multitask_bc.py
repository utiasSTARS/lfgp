import argparse
import numpy as np
import os
import torch

import rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state as p_aux
import rl_sandbox.constants as c
import rl_sandbox.examples.lfgp.experiment_utils as exp_utils
import rl_sandbox.transforms.general_transforms as gt

from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.absorbing_state import AbsorbingStateWrapper
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.train.train_multitask_bc import train_multitask_bc
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import MultiTaskFullyConnectedSquashedGaussianSAC
from rl_sandbox.model_architectures.layers_definition import VALUE_BASED_LINEAR_LAYERS


def str2tuple(v):
    return tuple([item for item in v.split(',')] if v else [])

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help="Random seed")
parser.add_argument('--user_machine', type=str, default='local', help="Representative string for user and machine")
parser.add_argument('--expert_paths', type=str2tuple, required=True, help="Comma-separated list of strings corresponding to the expert buffer files")
parser.add_argument('--exp_name', type=str, default="", help="String corresponding to the experiment name")
parser.add_argument('--main_task', type=str, default="stack_01", help="Main task (for play environment)")
parser.add_argument('--device', type=str, default="cpu", help="device to use")
parser.add_argument('--render', action='store_true', default=False, help="Render training")
parser.add_argument('--num_training', type=int, default=1, help="Number of training steps")
parser.add_argument('--num_updates', type=int, default=10000, help="Number of updates per training step")
parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
parser.add_argument('--num_evals_per_task', type=int, default=50, help="Number of evaluation episodes per task")
parser.add_argument('--num_overfit', type=int, default=100, help="Overfit tolerance")
parser.add_argument('--gpu_buffer', action='store_true', default=False, help="Store buffers on gpu.")
args = parser.parse_args()

seed = args.seed

save_path = exp_utils.get_save_path(c.MULTITASK_BC, args.main_task, args.seed, args.exp_name, args.user_machine)

for expert_path in args.expert_paths:
    assert os.path.isfile(expert_path), "File {} does not exist".format(expert_path)

aux_reward = p_aux.PandaPlayXYZStateAuxiliaryReward(args.main_task, include_main=False)
num_tasks = aux_reward.num_auxiliary_rewards
expert_buffers = args.expert_paths
num_evaluation_episodes = args.num_evals_per_task * num_tasks

obs_dim = 60  # +1 for absorbing state
action_dim = 4
min_action = -np.ones(action_dim)
max_action = np.ones(action_dim)
device = torch.device(args.device)

action_repeat = 1
num_frames = 1

memory_size = max_total_steps = args.num_training

max_episode_length = 360

buffer_settings = {
    c.KWARGS: {
        c.MEMORY_SIZE: memory_size,
        c.OBS_DIM: (obs_dim,),
        c.H_STATE_DIM: (1,),
        c.ACTION_DIM: (action_dim,),
        c.REWARD_DIM: (num_tasks,),
        c.INFOS: {c.MEAN: ((action_dim,), np.float32),
                    c.VARIANCE: ((action_dim,), np.float32),
                    c.ENTROPY: ((action_dim,), np.float32),
                    c.LOG_PROB: ((1,), np.float32),
                    c.VALUE: ((1,), np.float32),
                    c.DISCOUNTING: ((1,), np.float32)},
        c.CHECKPOINT_INTERVAL: 0,
        c.CHECKPOINT_PATH: None,
    },
    c.STORAGE_TYPE: c.RAM,
    c.STORE_NEXT_OBSERVATION: True,
    c.BUFFER_WRAPPERS: [
        {
            c.WRAPPER: TorchBuffer,
            c.KWARGS: {}
        },
    ]
}
if args.gpu_buffer:
    buffer_settings[c.KWARGS][c.DEVICE] = device
    buffer_settings[c.STORAGE_TYPE] = c.GPU
    buffer_settings[c.BUFFER_WRAPPERS] = []

experiment_setting = {
    # Auxiliary Tasks
    c.AUXILIARY_TASKS: {},

    # Buffer
    c.BUFFER_PREPROCESSING: gt.AsType(),
    c.BUFFER_SETTING: buffer_settings,

    # Environment
    c.ACTION_DIM: action_dim,
    c.CLIP_ACTION: True,
    c.ENV_SETTING: {
        c.ENV_BASE: {
            c.ENV_NAME: "PandaPlayInsertTrayXYZState",
        },
        c.KWARGS: {
            c.MAIN_TASK: args.main_task,
        },
        c.ENV_TYPE: c.MANIPULATOR_LEARNING,
        c.ENV_WRAPPERS: [
            {
                c.WRAPPER: AbsorbingStateWrapper,
                c.KWARGS: {
                    c.CREATE_ABSORBING_STATE: True,
                    c.MAX_EPISODE_LENGTH: max_episode_length,
                }
            },
            {
                c.WRAPPER: ActionRepeatWrapper,
                c.KWARGS: {
                    c.ACTION_REPEAT: action_repeat,
                    c.DISCOUNT_FACTOR: 1.,
                    c.ENABLE_DISCOUNTING: False,
                }
            },
            {
                c.WRAPPER: FrameStackWrapper,
                c.KWARGS: {
                    c.NUM_FRAMES: num_frames,
                }
            }
        ]
    },
    c.MIN_ACTION: min_action,
    c.MAX_ACTION: max_action,
    c.OBS_DIM: obs_dim,

    # Evaluation
    c.EVALUATION_FREQUENCY: 1,
    c.EVALUATION_RENDER: args.render,
    c.EVALUATION_RETURNS: [],
    c.NUM_EVALUATION_EPISODES: num_evaluation_episodes,
    
    # General
    c.DEVICE: device,
    c.SEED: seed,

    # Load
    c.LOAD_MODEL: False,

    # Logging
    c.PRINT_INTERVAL: 1,
    c.SAVE_INTERVAL: 1,
    c.LOG_INTERVAL: 1,

    # Model
    c.MODEL_SETTING: {
        c.MODEL_ARCHITECTURE: MultiTaskFullyConnectedSquashedGaussianSAC,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.TASK_DIM: num_tasks,
            c.ACTION_DIM: action_dim,
            c.SHARED_LAYERS: VALUE_BASED_LINEAR_LAYERS(in_dim=obs_dim),
            c.INITIAL_ALPHA: 1.,
            c.DEVICE: device,
            c.NORMALIZE_OBS: False,
            c.NORMALIZE_VALUE: False,
            c.BRANCHED_OUTPUTS: True
        },
    },
    
    c.OPTIMIZER_SETTING: {
        c.POLICY: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
    },

    c.EVALUATION_PREPROCESSING: gt.Identity(),
    c.TRAIN_PREPROCESSING: gt.Identity(),

    # BC
    c.NUM_TASKS: num_tasks,
    c.COEFFICIENTS: [1. for _ in range(num_tasks)],
    c.MULTI_BC_DATASET_SIZE_REWEIGHT: True,
    c.EXPERT_BUFFERS: expert_buffers,
    c.ACCUM_NUM_GRAD: 1,
    c.OPT_EPOCHS: args.num_updates,
    c.OPT_BATCH_SIZE: args.batch_size,
    c.MAX_GRAD_NORM: 1e10,
    c.VALIDATION_RATIO: 0.3,
    c.OVERFIT_TOLERANCE: args.num_overfit,

    c.AUXILIARY_REWARDS: aux_reward,

    # Progress Tracking
    c.CUM_EPISODE_LENGTHS: [0],
    c.CURR_EPISODE: 1,
    c.NUM_UPDATES: 0,
    c.RETURNS: [],

    # Save
    c.SAVE_PATH: save_path,

    # train parameters
    c.MAX_TOTAL_STEPS: max_total_steps,
    c.TRAIN_RENDER: False,
}

train_multitask_bc(experiment_config=experiment_setting)
