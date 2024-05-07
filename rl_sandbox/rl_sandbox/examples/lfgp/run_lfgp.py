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


def str2tuple(v):
    return tuple([item for item in v.split(',')] if v else [])

parser = argparse.ArgumentParser()

# RL
parser.add_argument('--seed', type=int, required=True, help="Random seed")
parser.add_argument('--device', type=str, default="cpu", help="device to use")
parser.add_argument('--render', action='store_true', default=False, help="Render training")
parser.add_argument('--max_steps', type=int, default=2000000, help="Number of steps to interact with")
parser.add_argument('--memory_size', type=int, default=4000000, help="Memory size of buffer")
parser.add_argument('--eval_freq', type=int, default=100000, help="The frequency of evaluating the performance of the current policy")
parser.add_argument('--num_evals_per_task', type=int, default=50, help="Number of evaluation episodes per task")
parser.add_argument('--log_interval', type=int, default=5000, help="Log interval for tensorboard.")
parser.add_argument('--buffer_warmup', type=int, default=25000, help="Buffer warmup before starting training.")
parser.add_argument('--exploration_steps', type=int, default=50000, help="Steps to use random instead of learned policy.")
parser.add_argument('--no_bootstrap_on_done', action="store_true", help="If set, use dones to prevent bootstrapping on timeouts.")
parser.add_argument('--no_entropy_in_qloss', action="store_true", help="If set, remove entropy from q loss.")
parser.add_argument('--debug_run', action="store_true", help="Drop log interval, buffer warmup, and exploration steps for debugging")

# env
parser.add_argument('--main_task', type=str, default="stack_01", help="Main task (for play environment)")
parser.add_argument('--main_intention', type=int, default=2, help="The main intention index")

# data
parser.add_argument('--top_save_path', type=str, default='results', help="Top directory for saving results")
parser.add_argument('--expert_paths', type=str2tuple, required=True, help="Comma-separated list of strings corresponding to the expert buffer files")
parser.add_argument('--exp_name', type=str, default="", help="String corresponding to the experiment name")
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

# n step
parser.add_argument('--n_step', type=int, default=1, help="If greater than 1, add an n-step loss to the q updates.")
parser.add_argument('--n_step_mode', type=str, default="n_rew_only",
                    help="N-step modes: options are: [n_rew_only, sum_pad, nth_q_targ].")
parser.add_argument('--nth_q_targ_multiplier', type=float, default=.5, help="applies to nth_q_targ n_step, .5 is value used in RCE.")

# lfgp/discriminator
parser.add_argument('--scheduler', type=str, default="wrs_plus_handcraft",
    help="Options: [wrs_plus_handcraft (default), wrs, learned, no_sched].")
parser.add_argument('--expbuf_last_sample_prop', type=float, default=0.95,
    help="Proportion of mini-batch samples that should be final transitions for discriminator training. 0.0 \
          means regular sampling.")
parser.add_argument('--expbuf_model_sample_rate', type=float, default=0.1,
    help="Proportion of mini-batch samples that should be expert samples for q/policy training.")
parser.add_argument('--expbuf_critic_share_type', type=str, default='share',
    help="Whether all critics learn from all expert buffers or from only their own. Options: [share, no_share]")
parser.add_argument('--expbuf_policy_share_type', type=str, default='share',
    help="Whether all policies learn from all expert buffers or from only their own. Options: [share, no_share]")
parser.add_argument('--expbuf_model_train_mode', type=str, default='both',
    help="Whether expert data trains the critic, or both the actor and critic. Options: [both, critic_only]")
parser.add_argument('--expbuf_model_sample_decay', type=float, default=0.99999,
    help="Decay rate for expbuf_model_sample_rate. .99999 brings close to 0 at 1M.")
parser.add_argument('--actor_raw_magnitude_penalty', type=float, default=0.0, help="L2 penalty on raw action (before tanh).")
parser.add_argument('--expert_data_mode', type=str, default="obs_act", help="options are [obs_act, obs_only, obs_only_no_next].")
parser.add_argument('--reward_model', type=str, default="discriminator", help="Options: [discriminator, sqil, rce]")
parser.add_argument('--sqil_rce_bootstrap_expert_mode', type=str, default="no_boot",
                    help="If boot, sqil and rce bootstrap on expert dones (unlike RCE implementation). no_boot"\
                         " means no bootstrapping on expert dones (but bootstrapping on non-expert handled by no_bootstrap_on_done)")
parser.add_argument('--q_type', type=str, default="raw", help="Options: [raw, classifier]")


args = parser.parse_args()

# make original lfgp readme consistent
if args.top_save_path == 'local': args.top_save_path = 'results'

# input sizes never change so this can/should give large speedup
torch.backends.cudnn.benchmark=True

if args.debug_run:
    args.buffer_warmup = 500
    args.exploration_steps = 1000

seed = args.seed

# for consistency with existing runs, we'll keep both wrs versions in lfgp_wrs
if args.scheduler == "wrs_plus_handcraft" or args.scheduler == "wrs":
    exp_name_dir = "lfgp_wrs"
else:
    exp_name_dir = "lfgp_" + str(args.scheduler)

save_path = exp_utils.get_save_path(exp_name_dir, args.main_task, args.seed, args.exp_name, args.top_save_path)

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

task_select_probs = [0.5 / (num_tasks - 1) for _ in range(num_tasks)]
task_select_probs[args.main_intention] = 0.5

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
                    c.DISCOUNTING: ((1,), np.float32),
                    c.VALUE: ((1,), np.float32),
                    c.HIGH_LEVEL_ACTION: ((1,), np.float32)},
        c.CHECKPOINT_INTERVAL: 0,
        c.CHECKPOINT_PATH: None,
        c.POLICY_SWITCH_DISCONTINUITY: False if args.n_step is None else True
    },
    c.STORAGE_TYPE: c.RAM,
    c.BUFFER_TYPE: c.STORE_NEXT_OBSERVATION if args.n_step is None else c.TRAJECTORY,
    c.BUFFER_WRAPPERS: [
        {
            c.WRAPPER: TorchBuffer,
            c.KWARGS: {}
        },
    ],
    c.LOAD_BUFFER: load_buffer,
}

if args.n_step:
    buffer_settings[c.KWARGS][c.N_STEP] = args.n_step

if args.gpu_buffer:
    buffer_settings[c.KWARGS][c.DEVICE] = device
    if args.n_step is None:
        buffer_settings[c.STORAGE_TYPE] = c.GPU
    else:
        buffer_settings[c.STORAGE_TYPE] = c.NSTEP_GPU
    buffer_settings[c.BUFFER_WRAPPERS] = []

# with no bootstrap on done, we have to remove final transitions, or things break
if args.no_bootstrap_on_done and args.n_step is not None:
    buffer_settings[c.KWARGS]["remove_final_transitions"] = True

# set scheduler
if args.scheduler == "wrs_plus_handcraft":
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
    else:
        handcraft_traj_options = [
            [4, 3, 2, 0, 4, 3, 2, 0],
            [4, 3, 5, 2, 0, 4, 3, 5],
            [3, 2, 0, 3, 2, 0, 3, 2],
            [2, 0, 2, 0, 2, 0, 2, 0],
            [5, 2, 0, 5, 2, 0, 5, 2],
        ]

    train_scheduler_setting = {
        c.MODEL_ARCHITECTURE: WeightedRandomSchedulerPlusHandcraft,
        c.KWARGS: {
            c.TASK_SELECT_PROBS: task_select_probs,
            c.MAX_SCHEDULE: math.ceil(max_episode_length / scheduler_period),
            c.NUM_TASKS: num_tasks,
            "handcraft_traj_epsilon": .5,
            "handcraft_traj_options": handcraft_traj_options,
        },
        c.SCHEDULER_PERIOD: scheduler_period,
    }

elif args.scheduler == "wrs":
    train_scheduler_setting = {
        c.MODEL_ARCHITECTURE: WeightedRandomScheduler,
        c.KWARGS: {
            c.TASK_SELECT_PROBS: task_select_probs,
            c.MAX_SCHEDULE: math.ceil(max_episode_length / scheduler_period),
            c.NUM_TASKS: num_tasks,
        },
        c.SCHEDULER_PERIOD: scheduler_period,
    }

elif args.scheduler == "learned":
    train_scheduler_setting = {
        c.MODEL_ARCHITECTURE: QTableScheduler,
        c.KWARGS: {
            c.MAX_SCHEDULE: math.ceil(max_episode_length / scheduler_period),
            c.NUM_TASKS: num_tasks,
            c.TEMPERATURE: 360.,
            c.TEMPERATURE_DECAY: 0.9995,
            c.TEMPERATURE_MIN: 0.1,
        },
        c.SCHEDULER_PERIOD: scheduler_period,
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

# discriminator obs only
disc_action_dim = 0 if 'obs_only' in args.expert_data_mode else action_dim


# TODO probably makes sense to remove these things once we know the configurations that work
# automatically set things for RCE and SQIL
if args.reward_model in ['sqil', 'rce']:
    if args.expbuf_model_sample_rate != 0.5:
        print('------------------')
        print(f"WARNING: Reward model {args.reward_model} but requested exp buf model rate of "\
              f"{args.expbuf_model_sample_rate}, setting to 0.5.")
        args.expbuf_model_sample_rate = 0.5
        print('------------------')
    if args.expbuf_model_sample_decay != 1.0:
        print('------------------')
        print(f"WARNING: Reward model {args.reward_model} but requested exp buf model decay of "\
              f"{args.expbuf_model_sample_decay}, setting to 1.0.")
        args.expbuf_model_sample_decay = 1.0
        print('------------------')
    if not args.no_entropy_in_qloss:
        print('------------------')
        print(f"WARNING: Entropy in q loss is set to on, turning off to match RCE.")
        args.no_entropy_in_qloss = True
        print('------------------')
    # if args.expbuf_critic_share_type != "no_share":
    #     print('------------------')
    #     print(f"WARNING: Expert buffer sampling should not be shared between tasks for RCE/SQIL. Setting to no_share.")
    #     args.expbuf_critic_share_type = "no_share"
    #     print('------------------')
    # if args.expbuf_policy_share_type != "no_share":
    #     print('------------------')
    #     print(f"WARNING: Expert buffer sampling should not be shared between tasks for RCE/SQIL. Setting to no_share.")
    #     args.expbuf_policy_share_type = "no_share"
    #     print('------------------')
    if args.expbuf_model_train_mode != "critic_only":
        print('------------------')
        print(f"WARNING: Expert buffer sampling should only train critic for RCE/SQIL. Setting to critic_only.")
        args.expbuf_model_train_mode = "critic_only"
        print('------------------')

if args.reward_model == 'rce':
    if args.q_type != "classifier":
        print('------------------')
        print(f"WARNING: Q type not set as classifier, setting to classifier to match RCE.")
        args.q_type = "classifier"
        print('------------------')
    if args.sqil_rce_bootstrap_expert_mode == 'boot':
        print('------------------')
        print(f"WARNING: sqil_rce_bootstrap_expert_mode set to 'boot', setting to 'no_boot' to match RCE settings.")
        args.sqil_rce_bootstrap_expert_mode = 'no_boot'
        print('------------------')

if args.q_type == 'classifier' and not args.no_entropy_in_qloss:
    print('------------------')
    print(f"WARNING: classifier q_type set, turning off entropy in qloss.")
    args.no_entropy_in_qloss = True
    print('------------------')


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
    c.EXPLORATION_STEPS: args.exploration_steps,
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
    c.LOG_INTERVAL: args.log_interval,

    # Model
    c.INTENTIONS_SETTING: {
        c.MODEL_ARCHITECTURE: MultiTaskFullyConnectedSquashedGaussianSAC,
        # c.MODEL_ARCHITECTURE: MultiTaskFullyConnectedSquashedGaussianTD3,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.TASK_DIM: num_tasks,
            c.ACTION_DIM: action_dim,
            c.SHARED_LAYERS: VALUE_BASED_LINEAR_LAYERS(in_dim=obs_dim),
            c.INITIAL_ALPHA: .01,  # for SAC only
            # c.POLICY_STDDEV: .7,  # for TD3 only
            # c.POLICY_STDDEV: [.7, .7, .7, .2],  # for TD3 only, per action dimension
            c.DEVICE: device,
            c.NORMALIZE_OBS: False,
            c.NORMALIZE_VALUE: False,
            c.BRANCHED_OUTPUTS: True,
            c.CLASSIFIER_OUTPUT: args.q_type == 'classifier',
        },
    },

    c.DISCRIMINATOR_SETTING: {
        c.MODEL_ARCHITECTURE: ActionConditionedFullyConnectedDiscriminator,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.ACTION_DIM: disc_action_dim,
            c.OUTPUT_DIM: num_tasks,
            c.LAYERS: SAC_DISCRIMINATOR_LINEAR_LAYERS(in_dim=obs_dim + disc_action_dim),
            c.DEVICE: device,
            c.OBS_ONLY: 'obs_only' in args.expert_data_mode,
        }
    },

    c.OPTIMIZER_SETTING: {
        c.INTENTIONS: {
            c.OPTIMIZER: torch.optim.AdamW,  # need this for proper weight decay
            c.KWARGS: {
                c.LR: 1e-5,
                # c.LR: 3e-4,
                c.WEIGHT_DECAY: 0.01,
            },
        },
        c.QS: {
            c.OPTIMIZER: torch.optim.AdamW,  # need this for proper weight decay
            c.KWARGS: {
                c.LR: 3e-4,
                c.WEIGHT_DECAY: 0.01,
            },
        },
        c.ALPHA: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: 3e-4,
            },
        },
        c.DISCRIMINATOR: {
            c.OPTIMIZER: torch.optim.AdamW,  # need this for proper weight decay
            c.KWARGS: {
                c.LR: 3e-4,
                c.WEIGHT_DECAY: 0.01,
            },
        },
    },

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

    c.EVALUATION_PREPROCESSING: gt.Identity(),
    c.TRAIN_PREPROCESSING: gt.Identity(),

    # LfGP
    c.EXPERT_BUFFERS: expert_buffers,
    c.DISCRIMINATOR_BATCH_SIZE: 256,
    c.GRADIENT_PENALTY_LAMBDA: 10.,
    c.DISCRIMINATOR_NUM_UPDATES: 1,
    c.DISCRIMINATOR_EXPBUF_LAST_SAMPLE_PROP: args.expbuf_last_sample_prop,
    c.EXPERT_BUFFER_MODEL_SAMPLE_RATE: args.expbuf_model_sample_rate,
    c.EXPERT_BUFFER_MODEL_SAMPLE_DECAY: args.expbuf_model_sample_decay,
    c.EXPERT_BUFFER_CRITIC_SHARE_ALL: args.expbuf_critic_share_type == 'share',  # all expert data used for all tasks w/ critic
    c.EXPERT_BUFFER_POLICY_SHARE_ALL: args.expbuf_policy_share_type == 'share',  # all expert data used for all tasks w/ policy
    c.EXPERT_BUFFER_MODEL_NO_POLICY: args.expbuf_model_train_mode == 'critic_only',
    c.REWARD_MODEL: args.reward_model,

    # SAC
    c.ACCUM_NUM_GRAD: 1,
    c.BATCH_SIZE: 256,
    c.BUFFER_WARMUP: args.buffer_warmup,
    c.GAMMA: 0.99,
    c.LEARN_ALPHA: True,
    c.MAX_GRAD_NORM: 10,
    c.NUM_GRADIENT_UPDATES: 1,
    c.NUM_PREFETCH: 1,
    c.REWARD_SCALING: 1.,
    c.STEPS_BETWEEN_UPDATE: 1,
    c.TARGET_ENTROPY: -float(action_dim),
    c.TARGET_UPDATE_INTERVAL: 1,
    c.TAU: 0.0001,
    c.UPDATE_NUM: 0,
    c.N_STEP: args.n_step,
    c.N_STEP_MODE: args.n_step_mode,
    c.NTH_Q_TARG_MULTIPLIER: args.nth_q_targ_multiplier,
    c.ACTOR_RAW_MAGNITUDE_PENALTY: args.actor_raw_magnitude_penalty,  # 0.0 to turn off
    c.BOOTSTRAP_ON_DONE: not args.no_bootstrap_on_done,
    c.EXPERT_DATA_MODE: args.expert_data_mode,
    c.SQIL_RCE_BOOTSTRAP_EXPERT_MODE: args.sqil_rce_bootstrap_expert_mode,
    c.NO_ENTROPY_IN_QLOSS: args.no_entropy_in_qloss,

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

exp_utils.config_check(experiment_setting, args.top_save_path)

train_lfgp_sac(experiment_config=experiment_setting)
