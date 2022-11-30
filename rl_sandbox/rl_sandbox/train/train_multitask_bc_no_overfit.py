import pickle
import gzip

import torch

import rl_sandbox.constants as c

from rl_sandbox.algorithms.bc.bc_no_overfit import MultitaskBC
from rl_sandbox.algorithms.sac_x.schedulers import FixedScheduler, RecycleScheduler
from rl_sandbox.auxiliary_tasks.utils import make_auxiliary_tasks
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.fake_env import FakeEnv
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import train
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.agents.hrl_agents import SACXAgent
from rl_sandbox.transforms.general_transforms import Identity
from rl_sandbox.utils import make_summary_writer, set_seed
from rl_sandbox.examples.lfgp.experts.subsample_expert_data import subsample_buffers

def train_multitask_bc_no_overfit(experiment_config):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())
    num_tasks = experiment_config[c.NUM_TASKS]

    set_seed(seed)
    train_env = FakeEnv(obs_dim=experiment_config[c.OBS_DIM])
    model = make_model(experiment_config[c.MODEL_SETTING])

    assert num_tasks == len(experiment_config[c.EXPERT_BUFFERS]) == experiment_config[c.AUXILIARY_REWARDS].num_auxiliary_rewards
    expert_buffers = []
    for load_path in experiment_config[c.EXPERT_BUFFERS]:
        # drop memory size for expert buffers to only what is needed
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)
            experiment_config[c.BUFFER_SETTING][c.KWARGS][c.MEMORY_SIZE] = data[c.MEMORY_SIZE]

        expert_buffers.append(make_buffer(experiment_config[c.BUFFER_SETTING], seed, load_path))

    if experiment_config.get(c.EXPERT_BUFFER_SUBSAMPLING, None) is not None:
        expert_buffers = subsample_buffers(expert_buffers, experiment_config[c.EXPERT_BUFFER_SUBSAMPLING])

    optimizer = make_optimizer(model.parameters(), experiment_config[c.OPTIMIZER_SETTING][c.POLICY])
    aux_tasks = make_auxiliary_tasks(experiment_config[c.AUXILIARY_TASKS],
                                     model,
                                     expert_buffers[0],
                                     experiment_config)

    learning_algorithm = MultitaskBC(model=model,
                                     optimizer=optimizer,
                                     expert_buffers=expert_buffers,
                                     algo_params=experiment_config,
                                     aux_tasks=aux_tasks)

    load_model = experiment_config.get(c.LOAD_MODEL, False)
    if load_model:
        learning_algorithm.load_state_dict(torch.load(load_model))

    agent = SACXAgent(scheduler=FixedScheduler(num_tasks=num_tasks,
                                               intention_i=0),
                      intentions=model,
                      learning_algorithm=learning_algorithm,
                      scheduler_period=c.MAX_INT,
                      preprocess=experiment_config[c.EVALUATION_PREPROCESSING])
    evaluation_env = None
    evaluation_agent = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0):
        assert experiment_config[c.NUM_EVALUATION_EPISODES] % num_tasks == 0
        evaluation_env = make_env(experiment_config[c.ENV_SETTING], seed + 1)
        evaluation_agent = SACXAgent(scheduler=RecycleScheduler(num_tasks=num_tasks,
                                                                scheduling=[experiment_config[c.NUM_EVALUATION_EPISODES] // num_tasks] * num_tasks),
                                     intentions=model,
                                     learning_algorithm=None,
                                     scheduler_period=c.MAX_INT,
                                     preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    summary_writer, save_path = make_summary_writer(save_path=save_path, algo=c.MULTITASK_BC, cfg=experiment_config)
    train(agent=agent,
          evaluation_agent=evaluation_agent,
          train_env=train_env,
          evaluation_env=evaluation_env,
          auxiliary_reward=experiment_config[c.AUXILIARY_REWARDS].reward,
          buffer_preprocess=buffer_preprocessing,
          experiment_settings=experiment_config,
          summary_writer=summary_writer,
          save_path=save_path)
