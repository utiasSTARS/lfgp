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
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import FullyConnectedSeparate
from rl_sandbox.model_architectures.layers_definition import VALUE_BASED_LINEAR_LAYERS
from rl_sandbox.train.train_bc import train_bc


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help="Random seed")
parser.add_argument('--top_save_path', type=str, default='results', help="Top directory for saving results")
parser.add_argument('--expert_path', type=str, required=True, help="String corresponding to the expert buffer file")
parser.add_argument('--exp_name', type=str, default="", help="String corresponding to the experiment name")
parser.add_argument('--main_task', type=str, default="stack_01", help="Main task (for play environment)")
parser.add_argument('--device', type=str, default="cpu", help="device to use")
parser.add_argument('--render', action='store_true', default=False, help="Render training")
parser.add_argument('--num_training', type=int, default=1, help="Number of training steps")
parser.add_argument('--num_updates', type=int, default=1000, help="Number of updates per training step")
parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
parser.add_argument('--num_evals', type=int, default=50, help="Number of evaluation episodes")
parser.add_argument('--num_overfit', type=int, default=100, help="Overfit tolerance")
parser.add_argument('--gpu_buffer', action='store_true', default=False, help="Store buffers on gpu.")
args = parser.parse_args()

# make original lfgp readme consistent
if args.top_save_path == 'local': args.top_save_path = 'results'

seed = args.seed

save_path = exp_utils.get_save_path(c.BC, args.main_task, args.seed, args.exp_name, args.top_save_path)

assert os.path.isfile(args.expert_path), "File {} does not exist".format(args.expert_path)

obs_dim = 60  # +1 for absorbing state
action_dim = 4
min_action = -np.ones(action_dim)
max_action = np.ones(action_dim)
device = torch.device(args.device)

action_repeat = 1
num_frames = 1

memory_size = max_total_steps = args.num_training

max_episode_length = 360

aux_reward_all = p_aux.PandaPlayXYZStateAuxiliaryReward(args.main_task, include_main=False)
aux_reward_names = [func.__qualname__ for func in aux_reward_all._aux_rewards]
# eval_reward = aux_reward_all._aux_rewards[aux_reward_names.index(args.main_task)]
eval_reward = aux_reward_all._aux_rewards[aux_reward_names.index("stack_0" if args.main_task == "unstack_stack_env_only_0" else args.main_task)]
# eval_reward = lambda reward, **kwargs: np.array([reward])  # use this for env reward

expert_buffer = args.expert_path
buffer_settings = {
    c.KWARGS: {
        c.MEMORY_SIZE: memory_size,
        c.OBS_DIM: (obs_dim,),
        c.H_STATE_DIM: (1,),
        c.ACTION_DIM: (action_dim,),
        c.REWARD_DIM: (1,),
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
    c.BUFFER_TYPE: c.STORE_NEXT_OBSERVATION,
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
            c.MAIN_TASK: args.main_task
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
    c.NUM_EVALUATION_EPISODES: args.num_evals,
    c.EVALUATION_REWARD_FUNC: eval_reward,

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
        c.MODEL_ARCHITECTURE: FullyConnectedSeparate,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.ACTION_DIM: action_dim,
            c.INITIAL_ALPHA: 1.,
            c.DEVICE: device,
            c.NORMALIZE_OBS: False,
            c.NORMALIZE_VALUE: False,
        },
    },

    c.OPTIMIZER_SETTING: {
        c.POLICY: {
            # c.OPTIMIZER: torch.optim.Adam,
            c.OPTIMIZER: torch.optim.AdamW,  # need this for proper weight decay
            c.KWARGS: {
                c.LR: 1e-5,
                c.WEIGHT_DECAY: 0.01,
            },
        },
    },

    c.EVALUATION_PREPROCESSING: gt.Identity(),
    c.TRAIN_PREPROCESSING: gt.Identity(),

    # BC
    c.ACCUM_NUM_GRAD: 1,
    c.OPT_EPOCHS: args.num_updates,
    c.OPT_BATCH_SIZE: args.batch_size,
    c.MAX_GRAD_NORM: 10,
    c.VALIDATION_RATIO: 0.3,
    c.OVERFIT_TOLERANCE: args.num_overfit,
    c.EXPERT_BUFFER: expert_buffer,

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

train_bc(experiment_config=experiment_setting)
