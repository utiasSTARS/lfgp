import numpy as np
import torch
import copy
import os
import ast

from rl_sandbox.agents.random_agents import UniformContinuousAgent, UniformContinuousActionRepeatAgent
import rl_sandbox.transforms.general_transforms as gt
from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer
from rl_sandbox.buffers.wrappers.noise_wrapper import NoiseBuffer
from rl_sandbox.envs.wrappers.action_repeat import ActionRepeatWrapper
from rl_sandbox.envs.wrappers.absorbing_state import AbsorbingStateWrapper
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.model_architectures.discriminators.fully_connected_discriminators import ActionConditionedFullyConnectedDiscriminator
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import \
    MultiTaskFullyConnectedSquashedGaussianSAC, MultiTaskFullyConnectedSquashedGaussianTD3
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import FullyConnectedSeparate
from rl_sandbox.model_architectures.layers_definition import VALUE_BASED_LINEAR_LAYERS, SAC_DISCRIMINATOR_LINEAR_LAYERS, \
                                                             CUSTOM_WIDTH_LINEAR_LAYERS
from rl_sandbox.envs.utils import make_env
import rl_sandbox.constants as c


def get_obs_action_dim(args):
    env_setting = get_env_settings(args)
    dummy_env = make_env(env_setting, dummy_env=True)
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    del dummy_env

    obs_dim *= args.frame_stack

    return obs_dim, action_dim

def main_task_alias_set(args):
    # renaming tasks with aliases to how they're used, allowing for more convenient arguments
    if args.main_task in ["stack", "bring", "insert", "move_obj", "lift", "reach"]:
        args.main_task = args.main_task + "_0"
    elif args.main_task in ["stack_no_move", "stack_nm"]:
        args.main_task = "stack_no_move_0"
    elif args.main_task in ["bring_no_move", "bring_nm"]:
        args.main_task = "bring_no_move_0"
    elif args.main_task in ["unstack_no_move", "unstack_nm"]:
        args.main_task = "unstack_stack_env_only_no_move_0"
    elif args.main_task in ["insert_no_bring_no_move", "insert_nb_nm"]:
        args.main_task = "insert_no_bring_no_move_0"
    elif args.main_task in ["insert_nb", "insert_no_bring"]:
        args.main_task = "insert_no_bring_0"
    elif args.main_task in ["unstack", "unstack-stack", "unstack_stack"]:
        args.main_task = "unstack_stack_env_only_0"
    elif args.main_task in ["unstack-with-us-aux", "unstack-stack-with-us-aux", "unstack_stack_with_us_aux"]:
        args.main_task = "unstack_stack_0"
    elif args.main_task in ["move"]:
        args.main_task = "move_obj_0"

    if not args.main_task.endswith("_0"):
        raise ValueError(f"Invalid main_task {args.main_task}")

def default_settings(args):
    # input sizes never change so this can/should give large speedup
    torch.backends.cudnn.benchmark=True

    # silence annoying tf warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.debug_run:
        args.buffer_warmup = 500
        args.exploration_steps = 1000
        args.log_interval = 100

def get_buffer_settings(args, obs_dim, action_dim, num_tasks, load_buffer, device):
    buffer_settings = {
        c.KWARGS: {
            c.MEMORY_SIZE: args.memory_size,
            c.OBS_DIM: (obs_dim,),
            c.H_STATE_DIM: (1,),
            c.ACTION_DIM: (action_dim,),
            c.REWARD_DIM: (num_tasks,),
            c.INFOS: {
                c.MEAN: ((action_dim,), np.float32),
                c.VARIANCE: ((action_dim,), np.float32),
                c.ENTROPY: ((action_dim,), np.float32),
                c.LOG_PROB: ((1,), np.float32),
                c.DISCOUNTING: ((1,), np.float32),
                c.VALUE: ((1,), np.float32)
            },
            c.CHECKPOINT_INTERVAL: 0,
            c.CHECKPOINT_PATH: None,
        },
        c.STORAGE_TYPE: c.RAM,
        c.BUFFER_TYPE: c.STORE_NEXT_OBSERVATION if args.n_step == 1 else c.TRAJECTORY,
        c.BUFFER_WRAPPERS: [
            {
                c.WRAPPER: TorchBuffer,
                c.KWARGS: {}
            },
        ],
        c.LOAD_BUFFER: load_buffer,
    }

    if num_tasks > 1:
        buffer_settings[c.KWARGS][c.INFOS][c.HIGH_LEVEL_ACTION] = ((1,), np.float32)

    if args.n_step > 1:
        buffer_settings[c.KWARGS][c.N_STEP] = args.n_step
        if num_tasks > 1:
            buffer_settings[c.KWARGS][c.POLICY_SWITCH_DISCONTINUITY] = True

    if args.gpu_buffer:
        buffer_settings[c.KWARGS][c.DEVICE] = device
        if args.n_step == 1:
            buffer_settings[c.STORAGE_TYPE] = c.GPU
        else:
            buffer_settings[c.STORAGE_TYPE] = c.NSTEP_GPU
        buffer_settings[c.BUFFER_WRAPPERS] = []

    # with no bootstrap on done, we have to remove final transitions, or things break
    if args.no_bootstrap_on_done and args.n_step > 1:
        buffer_settings[c.KWARGS]["remove_final_transitions"] = True

    expert_buffer_settings = copy.deepcopy(buffer_settings)
    if args.expert_randomize_factor > 0:
        expert_buffer_settings[c.BUFFER_WRAPPERS].append(
            {
                c.WRAPPER: NoiseBuffer,
                c.KWARGS: {
                    "noise_magnitude": args.expert_randomize_factor
                }
            }
        )

    if args.buffer_randomize_factor > 0:
        buffer_settings[c.BUFFER_WRAPPERS].append(
            {
                c.WRAPPER: NoiseBuffer,
                c.KWARGS: {
                    "noise_magnitude": args.buffer_randomize_factor,
                    "update_on_sample": True,
                }
            }
        )

    if args.exponential_sampling_method != "none":
        buffer_settings[c.KWARGS][c.EXPONENTIAL_SAMPLING_METHOD] = args.exponential_sampling_method
        buffer_settings[c.KWARGS][c.EXPONENTIAL_SAMPLING_PARAM] = args.exponential_sampling_param
        buffer_settings[c.KWARGS][c.EXPONENTIAL_UNIFORM_PROP] = args.exponential_uniform_prop

    return buffer_settings, expert_buffer_settings

def get_discriminator_settings(args, obs_dim, action_dim, num_tasks, device):
    # discriminator obs only
    disc_action_dim = 0 if 'obs_only' in args.expert_data_mode else action_dim

    discriminator_settings = {
        c.MODEL_ARCHITECTURE: ActionConditionedFullyConnectedDiscriminator,
        c.KWARGS: {
            c.OBS_DIM: obs_dim,
            c.ACTION_DIM: disc_action_dim,
            c.OUTPUT_DIM: num_tasks,
            c.SHARED_LAYERS: SAC_DISCRIMINATOR_LINEAR_LAYERS(in_dim=obs_dim + disc_action_dim),
            c.DEVICE: device,
            c.OBS_ONLY: 'obs_only' in args.expert_data_mode,
            c.ACTIVATION: getattr(torch.nn, args.disc_activation_func)()
        }
    }

    if not args.single_task:
        if args.no_disc_shared_layers:
            discriminator_settings[c.KWARGS][c.SHARED_LAYERS] = None
            discriminator_settings[c.KWARGS][c.BRANCHED_OUTPUTS] = True  # required no no shared layers

        if args.disc_branched_outputs:
            discriminator_settings[c.KWARGS][c.BRANCHED_OUTPUTS] = True

    return discriminator_settings

def str_to_kwargs(kwargs_str):
    kwargs = {}
    kwargs_combos = kwargs_str.split(',')
    for pair in kwargs_combos:
        key, arg = pair.split(':')
        arg = ast.literal_eval(arg)
        kwargs[key] = arg

    return kwargs

def get_env_settings(args):
    # environment
    env_setting = {
        c.ENV_WRAPPERS: [
            # {
            #     c.WRAPPER: AbsorbingStateWrapper,
            #     c.KWARGS: {
            #         c.CREATE_ABSORBING_STATE: True,
            #         c.MAX_EPISODE_LENGTH: args.max_episode_length,
            #     }
            # },
            {
                c.WRAPPER: ActionRepeatWrapper,
                c.KWARGS: {
                    c.ACTION_REPEAT: args.action_repeat,
                    c.DISCOUNT_FACTOR: 1.,
                    c.ENABLE_DISCOUNTING: False,
                }
            },
            {
                c.WRAPPER: FrameStackWrapper,
                c.KWARGS: {
                    c.NUM_FRAMES: args.frame_stack,
                }
            }
        ]
    }

    env_setting[c.ENV_TYPE] = args.env_type

    if args.env_type == 'manipulator_learning':
        env_setting[c.ENV_BASE] = {c.ENV_NAME: "PandaPlayInsertTrayXYZState"}
        env_setting[c.KWARGS] = {
            c.MAIN_TASK: args.main_task,
            "force_cube_rot_fix": True,
            "n_substeps": round(100 / args.env_control_hz),  # since env physics runs at 100Hz
            "max_real_time": args.env_max_real_time,
            "precalc_substeps": True,
            c.STATE_DATA: args.env_state_data.split(",")
        }

    elif args.env_type == c.SAWYER:
        env_setting[c.ENV_BASE] = {c.ENV_NAME: args.env_name}
        env_setting[c.KWARGS] = {
            "grip_pos_in_env": args.sawyer_grip_pos_in_env,
            "vel_in_env": args.sawyer_vel_in_env,
        }

    elif args.env_type == c.HAND_DAPG:
        env_setting[c.ENV_BASE] = {c.ENV_NAME: args.env_name}

        if 'dp' in args.env_name and args.hand_dapg_dp_kwargs != "":
            hand_dapg_dp_kwargs = {}
            kwargs_combos = args.hand_dapg_dp_kwargs.split(',')
            for pair in kwargs_combos:
                key, arg = pair.split(':')
                arg = ast.literal_eval(arg)
                hand_dapg_dp_kwargs[key] = arg

            env_setting[c.KWARGS] = {
                'hand_dapg_dp_kwargs': hand_dapg_dp_kwargs
            }

    elif args.env_type == c.PANDA_RL_ENVS:
        env_setting[c.ENV_BASE] = {c.ENV_NAME: args.env_name}


    return env_setting

def get_rl_settings(args, obs_dim, action_dim, num_evaluation_episodes):
    min_action = -np.ones(action_dim)
    max_action = np.ones(action_dim)
    max_total_steps = args.max_steps // args.action_repeat

    # environment
    env_setting = get_env_settings(args)

    # TODO add this as option
    exploration_strategy = UniformContinuousAgent(min_action, max_action, np.random.RandomState(args.seed))
    # exploration_strategy = UniformContinuousActionRepeatAgent(min_action, max_action, max_repeat=10, min_repeat=1,
    #                                                           rng=np.random.RandomState(args.seed))

    rl_settings = {
        # General
        c.SEED: args.seed,

        # Auxiliary Tasks
        c.AUXILIARY_TASKS: {},

        # Buffer
        c.BUFFER_PREPROCESSING: gt.AsType(),

        # Environment
        c.ACTION_DIM: action_dim,
        c.CLIP_ACTION: True,
        c.ENV_SETTING: env_setting,
        c.MIN_ACTION: min_action,
        c.MAX_ACTION: max_action,
        c.MAX_EPISODE_LENGTH: args.max_episode_length,
        c.OBS_DIM: obs_dim,
        c.TRAIN_DURING_ENV_STEP: args.train_during_env_step,

        # Evaluation
        c.NUM_EVALUATION_EPISODES: num_evaluation_episodes,
        c.EVALUATION_FREQUENCY: args.eval_freq,
        c.EVALUATION_RENDER: args.eval_render,
        c.EVALUATION_RETURNS: [],
        c.EVALUATION_STOCHASTIC: args.eval_mode == 'sto',

        # Exploration
        c.EXPLORATION_STEPS: args.exploration_steps,
        c.EXPLORATION_STRATEGY: exploration_strategy,

        # Logging
        c.PRINT_INTERVAL: args.print_interval,
        c.SAVE_INTERVAL: args.save_interval,
        c.LOG_INTERVAL: args.log_interval,
        c.CHECKPOINT_EVERY_EP: args.checkpoint_every_ep,
        c.LOAD_LATEST_CHECKPOINT: args.load_latest_checkpoint,
        c.CHECKPOINT_NAME: args.checkpoint_name,
        c.SAVE_CHECKPOINT_NAME: args.save_checkpoint_name,
        c.LOAD_BUFFER_NAME: args.load_buffer_name,
        c.LOAD_MODEL_NAME: args.load_model_name,
        c.LOAD_BUFFER_START_INDEX: args.load_buffer_start_index,

        # train parameters
        c.MAX_TOTAL_STEPS: max_total_steps,
        c.TRAIN_RENDER: args.render,
    }

    return rl_settings

def get_optimizer_settings(args):
    if args.single_task:
        policy_name = c.POLICY
    else:
        policy_name = c.INTENTIONS

    optimizizer_settings = {
        policy_name: {
            c.OPTIMIZER: torch.optim.AdamW,  # need this for proper weight decay
            c.KWARGS: {
                c.LR: args.actor_lr,
                c.WEIGHT_DECAY: args.p_weight_decay,
            },
        },
        c.QS: {
            c.OPTIMIZER: torch.optim.AdamW,  # need this for proper weight decay
            c.KWARGS: {
                c.LR: args.critic_lr,
                c.WEIGHT_DECAY: args.c_weight_decay,
            },
        },
        c.ALPHA: {
            c.OPTIMIZER: torch.optim.Adam,
            c.KWARGS: {
                c.LR: args.alpha_lr,
            },
        },
        c.DISCRIMINATOR: {
            c.OPTIMIZER: torch.optim.AdamW,  # need this for proper weight decay
            c.KWARGS: {
                c.LR: args.discriminator_lr,
                c.WEIGHT_DECAY: args.d_weight_decay,
            },
        },
    }

    return optimizizer_settings

def get_train_settings(args, action_dim, device):
    train_settings = {
        # general
        c.EVALUATION_PREPROCESSING: gt.Identity(),
        c.TRAIN_PREPROCESSING: gt.Identity(),
        c.DEVICE: device,

        # discriminator
        c.DISCRIMINATOR_BATCH_SIZE: args.batch_size,
        c.GRADIENT_PENALTY_LAMBDA: 10.,
        c.DISCRIMINATOR_NUM_UPDATES: 1,
        c.DISCRIMINATOR_EXPBUF_LAST_SAMPLE_PROP: args.expbuf_last_sample_prop,
        c.OBS_DIM_DISC_IGNORE: [int(x) for x in args.obs_dim_disc_ignore.split(',')] \
            if args.obs_dim_disc_ignore is not None else None,
        c.REW_MIN_ZERO: args.rew_min_zero,
        c.RMZ_NUM_MED_FILT: args.rmz_num_med_filt,

        # expert buffer sampling
        c.EXPERT_BUFFER_MODEL_SAMPLE_RATE: args.expbuf_model_sample_rate,
        c.EXPERT_BUFFER_MODEL_SAMPLE_DECAY: args.expbuf_model_sample_decay,
        c.EXPERT_BUFFER_CRITIC_SHARE_ALL: args.expbuf_critic_share_type == 'share',  # all expert data used for all tasks w/ critic
        c.EXPERT_BUFFER_POLICY_SHARE_ALL: args.expbuf_policy_share_type == 'share',  # all expert data used for all tasks w/ policy
        c.EXPERT_BUFFER_MODEL_NO_POLICY: args.expbuf_model_train_mode == 'critic_only',
        c.EXPERT_BUFFER_SIZE_TYPE: args.expbuf_size_type,
        c.EXPERT_CRITIC_WEIGHT: args.expert_critic_weight,

        # Discriminator/SQIL/RCE
        c.REWARD_MODEL: args.reward_model,

        # SAC
        c.ACCUM_NUM_GRAD: 1,
        c.BATCH_SIZE: args.batch_size,
        c.BUFFER_WARMUP: args.buffer_warmup,
        c.GAMMA: args.discount_factor,
        c.LEARN_ALPHA: args.sac_alpha_mode == 'learned',
        c.MAX_GRAD_NORM: 10,
        c.NUM_GRADIENT_UPDATES: args.num_gradient_updates,
        c.NUM_PREFETCH: 1,
        c.REWARD_SCALING: args.reward_scaling,
        c.STEPS_BETWEEN_UPDATE: 1,
        c.TARGET_ENTROPY: -float(action_dim),
        c.TARGET_UPDATE_INTERVAL: 1,
        c.TAU: args.target_polyak_averaging,
        c.UPDATE_NUM: 0,
        c.N_STEP: args.n_step,
        c.N_STEP_MODE: args.n_step_mode,
        c.NTH_Q_TARG_MULTIPLIER: args.nth_q_targ_multiplier,
        c.ACTOR_RAW_MAGNITUDE_PENALTY: args.actor_raw_magnitude_penalty,
        c.BOOTSTRAP_ON_DONE: not args.no_bootstrap_on_done,
        c.EXPERT_DATA_MODE: args.expert_data_mode,
        c.SQIL_RCE_BOOTSTRAP_EXPERT_MODE: args.sqil_rce_bootstrap_expert_mode,
        c.NO_ENTROPY_IN_QLOSS: args.no_entropy_in_qloss,

        # Progress Tracking
        c.CUM_EPISODE_LENGTHS: [0],
        c.CURR_EPISODE: 1,
        c.NUM_UPDATES: 0,
        c.RETURNS: [0],
    }

    return train_settings

def get_model_kwargs(args, obs_dim, action_dim, device):
    kwargs = {
        c.OBS_DIM: obs_dim,
        c.ACTION_DIM: action_dim,
        c.INITIAL_ALPHA: args.sac_initial_alpha,  # for SAC only
        # c.POLICY_STDDEV: .7,  # for TD3 only
        # c.POLICY_STDDEV: [.7, .7, .7, .2],  # for TD3 only, per action dimension
        c.DEVICE: device,
        c.NORMALIZE_OBS: args.obs_rms,
        c.NORMALIZE_VALUE: False,
        c.CLASSIFIER_OUTPUT: False
    }
    if not args.no_shared_layers:
        kwargs[c.SHARED_LAYERS] = CUSTOM_WIDTH_LINEAR_LAYERS(in_dim=obs_dim, width=args.shared_layers_width)
    if not args.single_task:
        kwargs[c.NUM_EXTRA_HIDDEN] = args.num_extra_hidden
        kwargs[c.TASK_SHARED_LAYERS_ONLY] = args.task_shared_layers_only
        if args.task_shared_layers_only:
            # override no_shared_layers
            kwargs[c.SHARED_LAYERS] = CUSTOM_WIDTH_LINEAR_LAYERS(in_dim=obs_dim, width=args.shared_layers_width)


    return kwargs
