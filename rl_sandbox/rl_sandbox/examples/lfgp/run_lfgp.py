import argparse
import math
import numpy as np
import torch

import rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state as p_aux
import rl_sandbox.constants as c
import rl_sandbox.examples.lfgp.experiment_utils as exp_utils
import rl_sandbox.examples.lfgp.transfer as transfer
import rl_sandbox.transforms.general_transforms as gt

from rl_sandbox.agents.random_agents import UniformContinuousAgent
from rl_sandbox.algorithms.sac_x.schedulers import QTableScheduler, RecycleScheduler
from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.absorbing_state import AbsorbingStateWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import MultiTaskFullyConnectedSquashedGaussianSAC
from rl_sandbox.model_architectures.discriminators.fully_connected_discriminators import ActionConditionedFullyConnectedDiscriminator
from rl_sandbox.model_architectures.layers_definition import VALUE_BASED_LINEAR_LAYERS, SAC_DISCRIMINATOR_LINEAR_LAYERS
from rl_sandbox.train.train_lfgp_sac import train_lfgp_sac


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
parser.add_argument('--max_steps', type=int, default=4000000, help="Number of steps to interact with")
parser.add_argument('--memory_size', type=int, default=4000000, help="Memory size of buffer")
parser.add_argument('--eval_freq', type=int, default=200000, help="The frequency of evaluating the performance of the current policy")
parser.add_argument('--num_evals_per_task', type=int, default=50, help="Number of evaluation episodes per task")
parser.add_argument('--main_intention', type=int, default=2, help="The main intention index")
parser.add_argument('--load_existing_dir', type=str, default="", help="dir with existing model, buffer, and exp settings.")
parser.add_argument('--load_model', type=str, default="", help="Path for model to be loaded")
parser.add_argument('--load_buffer', type=str, default="", help="Path for buffer to be loaded")
parser.add_argument('--load_transfer_exp_settings', type=str, default="",
                    help="The experimental settings of a previous run. If set, transfer any possible auxiliaries"
                         "from this old model to the new one.")
parser.add_argument('--load_max_buffer_index', type=int, required=False, help="If transferring, max buffer index.")
parser.add_argument('--load_aux_old_removal', type=str, required=False, default="", 
                     help="comma sep list of aux tasks from old model to ignore for transfer.")
parser.add_argument('--gpu_buffer', action='store_true', default=False, help="Store buffers on gpu.")

args = parser.parse_args()

seed = args.seed

save_path = exp_utils.get_save_path(c.LFGP, args.main_task, args.seed, args.exp_name, args.user_machine)

load_model, load_buffer, load_transfer_exp_settings, load_aux_old_removal = transfer.get_transfer_params(
    args.load_existing_dir, args.load_model, args.load_buffer, args.load_transfer_exp_settings, args.load_aux_old_removal)

aux_reward = p_aux.PandaPlayXYZStateAuxiliaryReward(args.main_task, include_main=False)
num_tasks = aux_reward.num_auxiliary_rewards
expert_buffers = args.expert_paths

obs_dim = 60  # +1 for absorbing state
action_dim = 4
min_action = -np.ones(action_dim)
max_action = np.ones(action_dim)
device = torch.device(args.device)

action_repeat = 1
num_frames = 1

memory_size = args.memory_size
max_total_steps = args.max_steps // action_repeat

max_episode_length = 360
scheduler_period = 45
num_evaluation_episodes = args.num_evals_per_task * num_tasks

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
    ],
    c.LOAD_BUFFER: load_buffer,
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
    c.MAX_EPISODE_LENGTH: max_episode_length,
    c.OBS_DIM: obs_dim,

    # Evaluation
    c.EVALUATION_FREQUENCY: args.eval_freq,
    c.EVALUATION_RENDER: args.render,
    c.EVALUATION_RETURNS: [],
    c.NUM_EVALUATION_EPISODES: num_evaluation_episodes,

    # Exploration
    c.EXPLORATION_STEPS: 1000,
    c.EXPLORATION_STRATEGY: UniformContinuousAgent(min_action,
                                                   max_action,
                                                   np.random.RandomState(seed)),
    
    # General
    c.DEVICE: device,
    c.SEED: seed,

    # Load
    c.LOAD_MODEL: load_model,
    c.LOAD_TRANSFER_EXP_SETTINGS: load_transfer_exp_settings,
    c.TRANSFER_PRETRAIN: 1000,
    c.TRANSFER_BUFFER_DOWNSAMPLE: 1.0,
    c.TRANSFER_BUFFER_MAX_INDEX: args.load_max_buffer_index,
    c.TRANSFER_AUX_IGNORE: load_aux_old_removal,

    # Logging
    c.PRINT_INTERVAL: 5000,
    c.SAVE_INTERVAL: 200000,
    c.LOG_INTERVAL: 5000,

    # Model
    c.INTENTIONS_SETTING: {
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

    c.DISCRIMINATOR_SETTING: {
        c.MODEL_ARCHITECTURE: ActionConditionedFullyConnectedDiscriminator,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.ACTION_DIM: action_dim,
            c.OUTPUT_DIM: num_tasks,
            c.LAYERS: SAC_DISCRIMINATOR_LINEAR_LAYERS(in_dim=obs_dim + action_dim),
            c.DEVICE: device,
        }
    },
    
    c.OPTIMIZER_SETTING: {
        c.INTENTIONS: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 1e-5,
            },
        },
        c.QS: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
        c.ALPHA: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
        c.DISCRIMINATOR: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
    },

    c.SCHEDULER_SETTING: {
        c.TRAIN: {
            c.MODEL_ARCHITECTURE: QTableScheduler,
            c.KWARGS: {
                c.MAX_SCHEDULE: math.ceil(max_episode_length / scheduler_period),
                c.NUM_TASKS: num_tasks,
                c.TEMPERATURE: 360.,
                c.TEMPERATURE_DECAY: 0.9995,
                c.TEMPERATURE_MIN: 0.1,
            },
            c.SCHEDULER_PERIOD: scheduler_period,
        },
        c.EVALUATION: {
            c.MODEL_ARCHITECTURE: RecycleScheduler,
            c.KWARGS: {
                c.NUM_TASKS: num_tasks,
                c.SCHEDULING: [num_evaluation_episodes // num_tasks] * num_tasks
            },
            c.SCHEDULER_PERIOD: c.MAX_INT,
        },
    },

    c.EVALUATION_PREPROCESSING: gt.Identity(),
    c.TRAIN_PREPROCESSING: gt.Identity(),

    # LfGP
    c.EXPERT_BUFFERS: expert_buffers,
    c.DISCRIMINATOR_BATCH_SIZE: 256,
    c.GRADIENT_PENALTY_LAMBDA: 10.,

    # SAC
    c.ACCUM_NUM_GRAD: 1,
    c.BATCH_SIZE: 256,
    c.BUFFER_WARMUP: 1000,
    c.GAMMA: 0.99,
    c.LEARN_ALPHA: True,
    c.MAX_GRAD_NORM: 10,
    c.NUM_GRADIENT_UPDATES: 1,
    c.NUM_PREFETCH: 1,
    c.REWARD_SCALING: 1.,
    c.STEPS_BETWEEN_UPDATE: 1,
    c.TARGET_ENTROPY: -float(action_dim),
    c.TARGET_UPDATE_INTERVAL: 1,
    c.TAU: 0.005,
    c.UPDATE_NUM: 0,

    # SACX
    c.AUXILIARY_REWARDS: aux_reward,
    c.NUM_TASKS: num_tasks,
    c.SCHEDULER_TAU: 0.6,
    c.MAIN_INTENTION: args.main_intention,

    # Progress Tracking
    c.CUM_EPISODE_LENGTHS: [0],
    c.CURR_EPISODE: 1,
    c.NUM_UPDATES: 0,
    c.RETURNS: [0],

    # Save
    c.SAVE_PATH: save_path,

    # train parameters
    c.MAX_TOTAL_STEPS: max_total_steps,
    c.TRAIN_RENDER: False,
}

train_lfgp_sac(experiment_config=experiment_setting)
