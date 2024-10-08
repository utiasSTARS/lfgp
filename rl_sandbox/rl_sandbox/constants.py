# ========================================================================
# General Terms
# ========================================================================

# Actions
CLIP_ACTION = "clip_action"
CONTINUOUS = "continuous"
DISCRETE = "discrete"
MAX_ACTION = "max_action"
MIN_ACTION = "min_action"
MIXED = "mixed"

# Algorithms
AC = "ac"
ALGAEDICE = "algaedice"
ALGO = "algo"
ALGORITHM = "algorithm"
ALGO_TYPE = "algo_type"
BASE = "base"
BC = "bc"
DAC = "dac"
LFGP = "lfgp"
LFGP_NS = "lfgp_ns"
DIAYN = "diayn"
DRQ = "drq"
GRAC = "grac"
MBPO = "mbpo"
MULTITASK_BC = "multitask_bc"
PPO = "ppo"
RCE = "rce"
SAC = "sac"
SACX = "sacx"
SAC_DIAYN = "sac_diayn"
SAC_DRQ = "sac_drq"
SAC_PER = "sac_per"
SAC_PT = "sac_pt"
VARIANT = "variant"

# Devices
CPU = "cpu"
DEVICE = "device"

# Dimensions
ACTION_DIM = "action_dim"
AXIS = "axis"
H_STATE_DIM = "h_state_dim"
HIDDEN_STATE_DIM = "hidden_state_dim"
IMG_DIM = "img_dim"
INPUT_DIM = "input_dim"
INPUT_SIZE = "input_size"
LATENT_DIM = "latent_dim"
LAYERS_DIM = "layers_dim"
OBS_DIM = "obs_dim"
OUTPUT_DIM = "output_dim"
REC_DIM = "rec_dim"
REWARD_DIM = "reward_dim"
STATE_DIM = "state_dim"
SCALAR_FEATURE_DIM = "scalar_feature_dim"
SHARED_OUT_DIM = "shared_out_dim"
TASK_DIM = "task_dim"
U_DIM = "u_dim"
Z_DIM = "z_dim"

# Real Robot additions
PANDA_RL_ENVS = "panda_rl_envs"
TRAIN_DURING_ENV_STEP = "train_during_env_step"
CHECKPOINT_EVERY_EP = "checkpoint_every_ep"
LOAD_LATEST_CHECKPOINT = "load_latest_checkpoint"
CHECKPOINT_NAME = "checkpoint_name"
SAVE_CHECKPOINT_NAME = "save_checkpoint_name"

# Environment
ABSORBING_STATE = "absorbing_state"
ACTION_REPEAT = "action_repeat"
DM_CONTROL = "dm_control"
DOMAIN_NAME = "domain_name"
ENV_BASE = "env_base"
ENV_NAME = "id"
ENV_SETTING = "env_setting"
ENV_TYPE = "env_type"
ENV_WRAPPERS = "env_wrappers"
GYM = "gym"
GYM_THING = "gym_thing"
MANIPULATOR_LEARNING = "manipulator_learning"
MAX_EPISODE_LENGTH = "max_episode_length"
OBS_TYPE = "obs_type"
ORIGINAL_OBS = "original_obs"
PIXELS = "pixels"
PIXELS_ONLY = "pixels_only"
RENDER = "render"
SAWYER = "sawyer"
HAND_DAPG = "hand_dapg"
STATE = "state"
TASK_KWARGS = "task_kwargs"
TASK_NAME = "task_name"
WRAPPER = "wrapper"
STATE_DATA = "state_data"

# General
COEFFICIENTS = "coefficients"
DONE_SUCCESS = "done_success"
GT = "gt"
IMG = "img"
KEYS = "keys"
KWARGS = "kwargs"
MAX_INT = 2 ** 32 - 1

# General Training
ACCUM_NUM_GRAD = "accum_num_grad"
BATCH_SIZE = "batch_size"
BUFFER_PREPROCESSING = "buffer_preprocessing"
EVALUATION = "evaluation"
EVALUATION_PREPROCESSING = "evaluation_preprocessing"
EVALUATION_IN_PARALLEL = "evaluation_in_parallel"
LEARNING_ALGORITHM = "learning_algorithm"
MAX_GRAD_NORM = "max_grad_norm"
NUM_GRADIENT_UPDATES = "num_gradient_updates"
NUM_UPDATES = "num_updates"
NUM_PREFETCH = "num_prefetch"
OPTIMIZER_SETTING = "optimizer_setting"
PREPROCESS = "preprocess"
REWARD_SCALING = "reward_scaling"
RNG = "rng"
OVERFIT_TOLERANCE = "overfit_tolerance"
STEPS_BETWEEN_UPDATE = "steps_between_update"
RANDOM = "random"
SEED = "seed"
TRAIN = "train"
TRAIN_PREPROCESSING = "train_preprocessing"
UPDATE_NUM = "update_num"
TORCH_RNG_STATE = "torch_rng_state"
NP_RNG_STATE = "np_rng_state"

# Graphics
RGB = "rgb"
RGB_ARRAY = "rgb_array"
WINDOW = "window"

# Gym Thing
IS_RENDER = "is_render"
RENDER_H = "render_h"
RENDER_W = "render_w"

# Hierarchical RL
HIGH_LEVEL_ACTION = "high_level_action"
HIGH_LEVEL_HIDDEN_STATE = "high_level_hidden_state"
HIGH_LEVEL_OBSERVATION = "high_level_observation"
NUM_TASKS = "num_tasks"
NUM_TRAIN_TASKS = "num_train_tasks"

# Image
HEIGHT = "height"
NUM_FRAMES = "num_frames"
NUM_IMAGES = "num_images"
WIDTH = "width"

# Losses
BC_LOSS = "bc_loss"
PG_LOSS = "pg_loss"
V_LOSS = "v_loss"

# Machine Learning
DECODER = "decoder"
ENCODER = "encoder"
ENTROPY = "entropy"
ENTROPIES = "entropies"
EPS = "eps"
HIDDEN_STATE = "hidden_state"
INITIALIZE_HIDDEN_STATE = "initialize_hidden_state"
LAYERS = "layers"
LR = "lr"
LOG_PROB = "log_prob"
MODEL_SETTING = "model_setting"
SHARED_LAYERS = "shared_layers"

# Manipulator Learning
OPEN = "open"
CLOSE = "close"

# Math Operations
REDUCTION = "reduction"
SUM = "sum"
P_NORM_EXP = "p_norm_exp"

# Model architectures
CNN = "cnn"
EARLY_FUSION_CNN = "early_fusion_cnn"
FC = "fc"
GRU = "gru"
LSTM = "lstm"
MODEL_ARCHITECTURE = "model_architecture"
MODEL_ARCHITECTURES = "model_architectures"
RNN = "rnn"
SEPARATE = "separate"
SHARED = "shared"

# Observation Normalization
NORMALIZE_OBS = "normalize_obs"
OBS_RMS = "obs_rms"

# Optimizer
ADAM = "adam"
RMSPROP = "rms_prop"
SGD = "sgd"
WEIGHT_DECAY = "weight_decay"

# Prioritized Experience Replay
PER = "per"
PER_ALPHA = "per_alpha"
PER_BETA = "per_beta"
PER_BETA_INCREMENT = "per_beta_increment"
PER_EPSILON = "per_epsilon"
IS_WEIGHT = "importance_sampling_weight"
PRIORITY = "priority"
SAMPLE_PROB = "sample_prob"
TD = "td"
TREE_IDX = "tree_idx"

# Prioritized Timestep Experience Replay
PT = "pt"
PRIORITY_DECAY = "priority_decay"
MIN_PRIORITY = "min_priority"

# PyTorch
BACKWARD = "backward"
OPTIMIZER = "optimizer"
STATE_DICT = "state_dict"

# Replay Buffer
ACTIONS = "actions"
BUFFER = "buffer"
BUFFER_SETTING = "buffer_setting"
EXPERT_BUFFER_SETTING = "expert_buffer_setting"
BUFFER_TYPE = "buffer_type"
BUFFER_WRAPPERS = "buffer_wrappers"
BURN_IN_WINDOW = "burn_in_window"
CHECKPOINT_INDICES = "checkpoint_idxes"
CHECKPOINT_INTERVAL = "checkpoint_interval"
CHECKPOINT_PATH = "checkpoint_path"
CIRCULAR = "circular"
COUNT = "count"
DEFAULT = "default"
DISK_DIR = "disk_dir"
DONES = "dones"
DONES_IDXES = "dones_idxes"
DTYPE = "dtype"
HIDDEN_STATES = "hidden_states"
HISTORY_FRAME = "history_frame"
HISTORY_LENGTH = "history_length"
INFOS = "infos"
MEMORY_SIZE = "memory_size"
NEXT_HIDDEN_STATES = "next_hidden_states"
NEXT_OBSERVATIONS = "next_observations"
OBSERVATIONS = "observations"
PADDING_FIRST = "padding_first"
POINTER = "pointer"
REWARDS = "rewards"
STORAGE_TYPE = "storage_type"
STORE_NEXT_OBSERVATION = "store_next_observation"
SUBDIR_LIMIT = "subdir_limit"
TRAJECTORY = "trajectory"
TMP_DIR = "./tmp"
POLICY_SWITCH_DISCONTINUITY = "policy_switch_discontinuity"

# Replay Buffer Type
BUFFER_TYPE = "buffer_type"
PRIORITIZED_EXPERIENCE_REPLAY = "prioritized_experience_replay"
VANILLA = "vanilla"

# RL general terms
STEP = "step"
AVERAGE_RETURNS = "average_returns"
CURR_EPISODE = "curr_episode"
CUM_EPISODE_LENGTHS = "cum_episode_lengths"
DISCOUNT_FACTOR = "discount_factor"
DISCOUNTING = "discounting"
ENABLE_DISCOUNTING = "enable_discounting"
EPISODE_IDXES = "episode_idxes"
EPISODE_LENGTHS = "episode_lengths"
EXPLORATION_STEPS = "exploration_steps"
EXPLORATION_STRATEGY = "exploration_strategy"
GAMMA = "gamma"
MAX_TOTAL_STEPS = "max_total_steps"
N_STEP = "n_step"
N_STEP_MODE = "n_step_mode"
NTH_Q_TARG_MULTIPLIER = "nth_q_targ_multiplier"
NUM_EPISODES = "num_episodes"
NUM_ITERS = "num_iters"
NUM_STEPS = "num_steps"
POLICY = "policy"
RETURNS = "returns"
SUCCESSES = "successes"
VALUE = "value"
BOOTSTRAP_ON_DONE = "bootstrap_on_done"

# RL transition tuple
ACTION = "action"
DONE = "done"
DONES = "dones"
INFO = "info"
OBSERVATION = "observation"
REWARD = "reward"
TRANSITION = "Transition"

# RNN
FLATTEN_PARAMETERS = "flatten_parameters"

# Statistics
GAUSSIAN = "gaussian"
MEAN = "mean"
STD = "std"
VAR = "var"
VARIANCE = "variance"
STANDARD_DEVIATION = "standard_deviation"

# Storage Type
DISK = "disk"
RAM = "ram"
GPU = "gpu"
NSTEP_GPU = "nstep_gpu"

# Testing
PUSH_TIME = "push_time"
REMOVE_TIME = "remove_time"

# Tracking
ACTOR_CRITIC_UPDATE_TIME = "actor_critic_update_time"
ADVANTAGE_COMPUTE_TIME = "advantage_compute_time"
ALPHA_UPDATE_TIME = "alpha_update_time"
CONTENT = "content"
LOAD_BUFFER = "load_buffer"
LOAD_MODEL = "load_model"
LOAD_BUFFER_NAME = "load_buffer_name"
LOAD_MODEL_NAME = "load_model_name"
LOAD_BUFFER_START_INDEX = "load_buffer_start_index"
LOAD_TRACKING_DICT = "load_tracking_dict"
LOG_SETTING = "log_setting"
MIN_MAX = "min_max"
MODEL_UPDATE_TIME = "model_update_time"
SAMPLE_TIME = "sample_time"
SAVE_PATH = "save_path"
POLICY_UPDATE_TIME = "policy_update_time"
Q_UPDATE_TIME = "q_update_time"
AGENT_UPDATE_TIME = "agent_update_time"
ENV_SAMPLE_TIME = "env_sample_time"
TARGET_UPDATE_TIME = "target_update_time"
UPDATE_INFO = "update_info"

# Transforms
CENTER_CROP = "center_crop"
RANDOM_CROP = "random_crop"
NORMALIZE = "normalize"

# ========================================================================
# Files specific
# ========================================================================

# exploration_strategy.py
GREEDY = "greedy"
UNIFORM_SAMPLE = "uniform_sample"

# learning_utils.py/train
EVALUATION_INFO = "evaluation_info"
INTERACTION_INFO = "interaction_info"
PRINT_INTERVAL = "print_interval"
SAVE_INTERVAL = "save_interval"
LOG_INTERVAL = "log_interval"
TERMINATION_BUFFER_FILE = "termination_buffer.pkl"
TERMINATION_STATE_DICT_FILE = "termination_state_dict.pt"
TERMINATION_TRAIN_FILE = "termination_train.pkl"
TRAIN_FILE = "train.pkl"
TRAIN_RENDER = "train_render"

# learning_utils.py/evaluate_policy
EVALUATION_FREQUENCY = "evaluation_frequency"
EVALUATION_RENDER = "evaluation_render"
EVALUATION_RETURNS = "evaluation_returns"
EVALUATION_SUCCESSES = "evaluation_successes"
EVALUATION_SUCCESSES_ALL_TASKS = "evaluation_successes_all_tasks"
NUM_EVALUATION_EPISODES = "num_evaluation_episodes"
EVALUATION_REWARD_FUNC = "evaluation_reward_func"
AVERAGE_MULTITASK_RETURNS = "average_multitask_returns"
EVALUATION_STOCHASTIC = "evaluation_stochastic"

# ========================================================================
# Training algorithms
# ========================================================================

# Auxiliary Tasks
AUXILIARY_TASKS = "auxiliary_tasks"
LOSS_COEF = "loss_coef"

# AlgaeDICE
ALGAE_ALPHA = "algae_alpha"
CRITIC_MIXTURE_RATIO = "critic_mixture_ratio"

# Behavior Cloning
MAX_EPOCHS = "max_epochs"
MAX_EPOCHS_WITHOUT_BEST = "max_epochs_without_best"
EPOCH_TRAIN_LOSS = "epoch_train_loss"
EPOCH_TRAIN_TIME = "epoch_train_time"
EPOCH_VALID_TIME = "epoch_valid_time"
MULTI_BC_DATASET_SIZE_REWEIGHT = "multi_bc_dataset_size_reweight"

# CEM
ELITE_SIZE = "elite_size"
POP_SIZE = "pop_size"

# DAC
CREATE_ABSORBING_STATE = "create_absorbing_state"
DISCRIMINATOR_BATCH_SIZE = "discriminator_batch_size"
DISCRIMINATOR_REWARD = "discriminator_reward"
DISCRIMINATOR_MAX = "discriminator_max"
DISCRIMINATOR_MIN = "discriminator_min"
DISCRIMINATOR_SAMPLE_TIME = "discriminator_sample_time"
DISCRIMINATOR_NUM_UPDATES = "discriminator_num_updates"
DISCRIMINATOR_EXPBUF_LAST_SAMPLE_PROP = "discriminator_expbuf_last_sample_prop"
DISCRIMINATOR_OBS_ONLY = "discriminator_obs_only"
EXPERT_BUFFER = "expert_buffer"
EXPERT_BUFFERS = "expert_buffers"
FT_EXPERT_BUFFER = "ft_expert_buffer"
FT_EXPERT_BUFFERS = "ft_expert_buffers"
EXPERT_AMOUNTS = "expert_amounts"
EXPERT_AMOUNT = "expert_amount"
EXPERT_SETTING = "expert_setting"
EXPERT_BUFFER_SUBSAMPLING = "expert_buffer_subsampling"
GAN_LOSS = "gan_loss"
GP_LOSS = "gp_loss"
GRADIENT_PENALTY_LAMBDA = "gradient_penalty_lambda"
EXPERT_BUFFER_MODEL_SAMPLE_RATE = "expert_buffer_model_sample_rate"
EXPERT_BUFFER_MODEL_SAMPLE_DECAY = "expert_buffer_model_sample_decay"
EXPERT_BUFFER_CRITIC_SHARE_ALL = "expert_buffer_critic_share_all"
EXPERT_BUFFER_POLICY_SHARE_ALL = "expert_buffer_policy_share_all"
EXPERT_BUFFER_SIZE_TYPE = "expert_buffer_size_type"
EXPERT_BUFFER_MODEL_NO_POLICY = "expert_buffer_model_no_policy"
OBS_ONLY = "obs_only"
EXPERT_DATA_MODE = "expert_data_mode"
EXPONENTIAL_SAMPLING_MED_MULT = "exponential_sampling_med_mult"
EXPONENTIAL_SAMPLING_MED_FIXED = "exponential_sampling_med_fixed"
EXPONENTIAL_SAMPLING_METHOD = "exponential_sampling_method"
EXPONENTIAL_SAMPLING_PARAM = "exponential_sampling_param"
EXPONENTIAL_UNIFORM_PROP = "exponential_uniform_prop"
OBS_DIM_DISC_IGNORE = "obs_dim_disc_ignore"
ACTIVATION = "activation"
REW_MIN_ZERO = "rew_min_zero"
RMZ_NUM_MED_FILT = "rmz_num_med_filt"

# SQIL/RCE
REWARD_MODEL = "reward_model"
SQIL = "sqil"
SPARSE = "sparse"
Q_REGULARIZER = "q_regularizer"
SQIL_RCE_BOOTSTRAP_EXPERT_MODE = "sqil_rce_bootstrap_expert_mode"
NO_ENTROPY_IN_QLOSS = "no_entropy_in_qloss"
CLASSIFIER_OUTPUT = "classifier_output"
RCE_EPS = "rce_eps"
AVG_EXPERT_Q = "avg_expert_q"
MAX_EXPERT_Q = "max_expert_q"
AVG_POLICY_Q = "avg_policy_q"
AVG_DONE_Q = "avg_done_q"
MAX_POLICY_Q = "max_policy_q"
Q_OVER_MAX_PENALTY = "q_over_max_penalty"
QOMP_NUM_MED_FILT = "qomp_num_med_filt"
QOMP_POLICY_MAX_TYPE = "qomp_policy_max_type"
Q_EXPERT_TARGET_MODE = "q_expert_target_mode"
NOISE_ZERO_TARGET_MODE = "noise_zero_target_mode"
NZT_PER_OBS_SCALE = "nzt_per_obs_scale"
EXPERT_CRITIC_WEIGHT = "expert_critic_weight"
SQIL_POLICY_REWARD_LABEL = "sqil_policy_reward_label"

# LfGP
MAIN_TASK = "main_task"
HANDCRAFT_REWARDS = "handcraft_rewards"
TASK_SHARED_LAYERS_ONLY = "task_shared_layers_only"
MAIN_TASK_LOSS_WEIGHT = "main_task_loss_weight"
NUM_EXTRA_HIDDEN = "num_extra_hidden"

# DIAYN
DISCRIMINATOR = "discriminator"
DISCRIMINATOR_LOSS = "discriminator_loss"
DISCRIMINATOR_OPTIMIZER = "discriminator_optimizer"
DISCRIMINATOR_SETTING = "discriminator_setting"
DISCRIMINATOR_UPDATE_TIME = "discriminator_update_time"

KL_APPROXIMATION_SAMPLES = "kl_approximation_samples"
PRIOR = "prior"
TASK = "task"

# DrQ
K = "K"
M = "M"

# GAE
GAE_LAMBDA = "gae_lambda"

# GRAC
AVG_Q1_VAL = "average_q1_value"
AVG_Q2_VAL = "average_q2_value"
AVG_Q_DISCREPANCY = "average_q_discrepancy"
CEM = "cem"
COV_NOISE_END = "cov_noise_end"
COV_NOISE_INIT = "cov_noise_init"
COV_NOISE_TAU = "cov_noise_tau"
LPROB_MAX = "lprob_max"
LPROB_MIN = "lprob_min"
NUM_Q_UPDATES = "num_q_updates"
Q1_MAX = "q1_max"
Q2_MAX = "q2_max"
Q1_REG = "q1_reg"
Q2_REG = "q2_reg"

# Initial exploration Strategy
UNIFORM = "uniform"

# Koopman
KOOPMAN = "koopman"
KOOPMAN_DYNAMICS = "koopman_dynamics"
KOOPMAN_OPTIMIZER = "koopman_optimizer"

# MBPO
VALIDATION_RATIO = "validation_ratio"
VALIDATION_LOSS = "validation_loss"
VALIDATION_SAMPLE_TIME = "validation_sample_time"

# Model-based
DYNAMICS = "dynamics"
MODEL_CONSTRUCTOR = "model_constructor"
MODEL_KWARGS = "model_kwargs"
MODEL_LOSS = "model_loss"
MODEL_NUM_GRADIENT_UPDATES = "model_num_gradient_updates"
MODEL_OPTIMIZER = "model_optimizer"
MODEL_STATE_DICT = "model_state_dict"
MODEL_VALIDATION_LOSS = "model_validation_loss"
MODEL_ROLLOUT_TIME = "model_rollout_time"
NUM_MODELS = "num_models"

# Off-policy
BUFFER_WARMUP = "buffer_warmup"
TARGET_UPDATE_INTERVAL = "target_update_interval"

# Polyak averaging
TAU = "tau"

# PopArt
NORMALIZE_VALUE = "normalize_value"
VALUE_RMS = "value_rms"

# PPO
CLIP_PARAM = "clip_param"
CLIP_VALUE = "clip_value"
ENT_COEF = "ent_coef"
NORMALIZE_ADVANTAGE = "normalize_advantage"
OPT_BATCH_SIZE = "opt_batch_size"
OPT_EPOCHS = "opt_epochs"
PG_COEF = "pg_coef"
V_COEF = "v_coef"

# Q-Learning
Q_LOSS = "q_loss"
Q_OPTIMIZER = "q_optimizer"
Q_TABLE = "q_table"

# RCE
CLASSIFIER = "classifier"
CLASSIFIER_BATCH_SIZE = "classifier_batch_size"
CLASSIFIER_OPTIMIZER = "classifier_optimizer"
CLASSIFIER_UPDATE_TIME = "classifier_update_time"
CLASSIFIER_SAMPLE_TIME = "classifier_sample_time"
CLASSIFIER_SETTING = "classifier_setting"
RCE_LOSS = "rce_loss"

# SAC
ACTOR_UPDATE_INTERVAL = "actor_update_interval"
ALPHA = "alpha"
ALPHA_LOSS = "alpha_loss"
ALPHA_OPTIMIZER = "alpha_optimizer"
INITIAL_ALPHA = "initial_alpha"
LEARN_ALPHA = "learn_alpha"
PI_LOSS = "pi_loss"
POLICY_OPTIMIZER = "pi_optimizer"
Q1_LOSS = "q1_loss"
Q2_LOSS = "q2_loss"
QS = "qs"
QS_OPTIMIZER = "qs_optimizer"
TARGET_ENTROPY = "target_entropy"
MAGNITUDE_PENALTY_MULTIPLIER = "magnitude_penalty_multiplier"
MAGNITUDE_MAX = "magnitude_max"
MAGNITUDE_FOLLOW_AVERAGE = "magnitude_follow_average"
ACTOR_RAW_MAGNITUDE_PENALTY = "actor_raw_magnitude_penalty"

# TD3/DDPG
POLICY_STDDEV = "policy_stddev"
TARGET_STDDEV = "target_stddev"
TARGET_STDDEV_CLIP = "target_stddev_clip"

# SAC-X
AUXILIARY_REWARDS = "auxiliary_rewards"
HANDCRAFT_TASKS = "handcraft_tasks"
INTENTIONS = "intentions"
INTENTION_I = "intention_i"
INTENTIONS_ALGO = "intentions_algo"
INTENTIONS_SETTING = "intentions_setting"
INTENTIONS_UPDATE_TIME = "intentions_update_time"
MAIN_INTENTION = "main_intention"
MAX_SCHEDULE = "max_schedule"
SCHEDULER = "scheduler"
SCHEDULER_PERIOD = "scheduler_period"
SCHEDULER_SETTING = "scheduler_setting"
SCHEDULER_TEMPERATURE = "scheduler_temperature"
SCHEDULER_TEMPERATURE_DECAY = "scheduler_temperature_decay"
SCHEDULER_TEMPERATURE_MIN = "scheduler_temperature_min"
SCHEDULER_TAU = "scheduler_tau"
SCHEDULER_UPDATE_TIME = "scheduler_update_time"
SCHEDULER_TRAJ = "scheduler_traj"
SCHEDULER_TRAJ_VALUE = "scheduler_traj_value"
SCHEDULING = "scheduling"
TEMPERATURE = "temperature"
TEMPERATURE_DECAY = "temperature_decay"
TEMPERATURE_MIN = "temperature_min"
BRANCHED_OUTPUTS = "branched_outputs"
LOAD_TRANSFER_EXP_SETTINGS = "load_transfer_exp_settings"
TRANSFER_PRETRAIN = "transfer_pretrain"
TRANSFER_BUFFER_DOWNSAMPLE = "transfer_buffer_downsample"
TRANSFER_BUFFER_MAX_INDEX = "transfer_buffer_max_index"
TRANSFER_AUX_IGNORE = "transfer_aux_ignore"
TASK_CONDITIONAL_PROBS = "task_conditional_probs"
TASK_RESET_PROBS = "task_reset_probs"
TASK_SELECT_PROBS = "task_select_probs"

# ========================================================================
# Training parameters
# ========================================================================

# Default Training
DEFAULT_TRAIN_PARAMS = {
    # Action
    CLIP_ACTION: True,
    MIN_ACTION: -1.,
    MAX_ACTION: 1.,

    # Evaluation
    EVALUATION_FREQUENCY: 5000,
    EVALUATION_RENDER: True,
    EVALUATION_RETURNS: [],
    EVALUATION_SUCCESSES: [],
    EVALUATION_SUCCESSES_ALL_TASKS: [],
    NUM_EVALUATION_EPISODES: 1,

    # Logging
    CHECKPOINT_INTERVAL: 10000,
    CHECKPOINT_PATH: 'checkpoints/',
    PRINT_INTERVAL: 100,
    SAVE_INTERVAL: 5,
    LOG_INTERVAL: 100,

    # Progress Tracking
    CUM_EPISODE_LENGTHS: [0],
    CURR_EPISODE: 1,
    NUM_UPDATES: 0,
    RETURNS: [],

    # Train
    MAX_TOTAL_STEPS: 1000000,
    TRAIN_RENDER: False,
}

# AlgaeDICE
DEFAULT_ALGAEDICE_PARAMS = {
    ALGAE_ALPHA: 0.01,
    CRITIC_MIXTURE_RATIO: 0.05,
    P_NORM_EXP: 2,
}

# BC
DEFAULT_BC_PARAMS = {
    ACCUM_NUM_GRAD: 1,
    OPT_BATCH_SIZE: 256,
    OPT_EPOCHS: 1,
    STEPS_BETWEEN_UPDATE: 1000,
    MAX_GRAD_NORM: 1e10,
    VALIDATION_RATIO: 0.3,
    MAX_EPOCHS: 1000,
    MAX_EPOCHS_WITHOUT_BEST: 30,
    OVERFIT_TOLERANCE: 30,
}

# DIAYN
DEFAULT_DIAYN_PARAMS = {
    KL_APPROXIMATION_SAMPLES: 100,
}

# DrQ
DEFAULT_DRQ_PARAMS = {
    M: 2,
    K: 2,
}

# GRAC
DEFAULT_GRAC_PARAMS = {
    ACCUM_NUM_GRAD: 1,
    ALPHA: 0.7,
    BATCH_SIZE: 256,
    BUFFER_TYPE: VANILLA,
    BUFFER_WARMUP: 1000,
    COV_NOISE_END: 0.05,
    COV_NOISE_INIT: 0.1,
    COV_NOISE_TAU: 0.95,
    ELITE_SIZE: 5,
    EXPLORATION_STEPS: 1000,
    EXPLORATION_STRATEGY: UNIFORM,
    GAMMA: 0.99,
    MIN_ACTION: -1,
    MAX_ACTION: 1,
    MAX_GRAD_NORM: 1.,
    MEMORY_SIZE: 1000000,
    NUM_GRADIENT_UPDATES: 1,
    NUM_ITERS: 2,
    NUM_Q_UPDATES: 4,
    POP_SIZE: 20,
    REWARD_SCALING: 1.,
    STEPS_BETWEEN_UPDATE: 1,
    UPDATE_NUM: 0,
}

# MBPO
DEFAULT_MBPO_PARAMS = {
    ACCUM_NUM_GRAD: 1,
    BATCH_SIZE: 64,
    VALIDATION_RATIO: 0.2,
    BUFFER_WARMUP: 1000,
    NUM_GRADIENT_UPDATES: 1,
    NUM_PREFETCH: 1,
    MAX_GRAD_NORM: 1e10,
    STEPS_BETWEEN_UPDATE: 1000,
    M: 400,
    K: 1,
}

# PER
DEFAULT_PER_PARAMS = {
    PER_ALPHA: 0.6,
    PER_BETA: 0.4,
    PER_BETA_INCREMENT: 0.001,
    PER_EPSILON: 1e-5,
}

# PPO
DEFAULT_PPO_PARAMS = {
    ACCUM_NUM_GRAD: 1,
    CLIP_PARAM: 0.02,
    CLIP_VALUE: True,
    ENT_COEF: 0.0,
    GAE_LAMBDA: 0.95,
    GAMMA: 0.99,
    MAX_GRAD_NORM: 1.,
    NORMALIZE_ADVANTAGE: True,
    OPT_BATCH_SIZE: 128,
    OPT_EPOCHS: 10,
    PG_COEF: 1.,
    STEPS_BETWEEN_UPDATE: 1024, # same as batch size
    V_COEF: 1.,
}

# PT
DEFAULT_PT_PARAMS = {
    PRIORITY_DECAY: 0.9,
    MIN_PRIORITY: 1e-5,
}

# RCE
DEFAULT_RCE_PARAMS = {
    EPS: 1e-5,
    N_STEP: 1,
}

# REINFORCE
DEFAULT_REINFORCE_PARAMS = {
    ACCUM_NUM_GRAD: 1,
    GAMMA: 0.99,
    MAX_GRAD_NORM: 1.,
    STEPS_BETWEEN_UPDATE: 1024,
}

# SAC
DEFAULT_SAC_PARAMS = {
    ACCUM_NUM_GRAD: 1,
    ACTOR_UPDATE_INTERVAL: 1,
    BATCH_SIZE: 256,
    BUFFER_TYPE: VANILLA,
    BUFFER_WARMUP: 1000,
    EXPLORATION_STEPS: 1000,
    EXPLORATION_STRATEGY: UNIFORM,
    GAMMA: 0.99,
    INITIAL_ALPHA: 1.,
    LEARN_ALPHA: True,
    MAX_GRAD_NORM: 1.,
    MEMORY_SIZE: 1000000,
    NUM_GRADIENT_UPDATES: 1,
    REWARD_SCALING: 1.,
    STEPS_BETWEEN_UPDATE: 1,
    TARGET_ENTROPY: 0.,
    TARGET_UPDATE_INTERVAL: 1,
    TAU: 0.005,
    UPDATE_NUM: 0,
}

# SAC-X
DEFAULT_SACX_PARAMS = {
    SCHEDULER_PERIOD: 500,
    SCHEDULER_TAU: 0.4,
    SCHEDULER_TEMPERATURE: 10.,
    SCHEDULER_TEMPERATURE_DECAY: 0.9999,
    SCHEDULER_TEMPERATURE_MIN: 1.,
}

# ========================================================================
# Valid settings
# ========================================================================

VALID_ACTION_TYPE = (CONTINUOUS,
                     DISCRETE,
                     MIXED,)

VALID_ALGORITHMS = (PPO,
                    SAC,
                    SACX,)

VALID_BUFFER_TYPE = (DEFAULT,
                     STORE_NEXT_OBSERVATION,
                     TRAJECTORY)

VALID_ENV_TYPE = (DM_CONTROL,
                  GYM,
                  GYM_THING,
                  MANIPULATOR_LEARNING,
                  SAWYER,
                  HAND_DAPG,
                  PANDA_RL_ENVS)

VALID_DISCRIMINATOR = (GAUSSIAN,)

VALID_ER_VARIANT = (VANILLA,
                    PER,)

VALID_HRL_ALGORITHMS = (SACX,)

VALID_INIT_EXPLORATION = (UNIFORM,)

VALID_INTENTIONS_ALGO = (SAC,)

VALID_OBS_TYPE = (GT,
                  IMG,)

VALID_OPTIMIZER = (ADAM,
                   RMSPROP,
                   SGD,)

VALID_PER_VARIANT = (PER,
                     PT,)

VALID_PRIOR = (GAUSSIAN,
               UNIFORM,)

VALID_RNN_TYPE = (GRU,
                  LSTM,
                  RNN,)

VALID_SAC_VARIANT = (VANILLA,
                     DIAYN,
                     DRQ,
                     PER,
                     PT)

VALID_SACX_ALGO = (SAC,)

VALID_STORAGE_TYPE = (DISK,
                      RAM,)
