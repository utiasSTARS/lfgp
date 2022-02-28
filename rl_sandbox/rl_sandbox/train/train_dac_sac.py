import torch

import rl_sandbox.constants as c

from rl_sandbox.algorithms.dac.sac import SACDAC
from rl_sandbox.algorithms.dac.dac import DAC
from rl_sandbox.auxiliary_tasks.utils import make_auxiliary_tasks
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import train
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.agents.rl_agents import ACAgent
from rl_sandbox.transforms.general_transforms import Identity
from rl_sandbox.utils import make_summary_writer, set_seed

def train_dac_sac(experiment_config):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())

    set_seed(seed)
    train_env = make_env(experiment_config[c.ENV_SETTING], seed)
    model = make_model(experiment_config[c.MODEL_SETTING])
    buffer = make_buffer(experiment_config[c.BUFFER_SETTING], seed, experiment_config[c.BUFFER_SETTING].get(c.LOAD_BUFFER, False))

    policy_opt = make_optimizer(model.policy_parameters, experiment_config[c.OPTIMIZER_SETTING][c.POLICY])
    qs_opt = make_optimizer(model.qs_parameters, experiment_config[c.OPTIMIZER_SETTING][c.QS])
    alpha_opt = make_optimizer([model.log_alpha], experiment_config[c.OPTIMIZER_SETTING][c.ALPHA])

    aux_tasks = make_auxiliary_tasks(experiment_config[c.AUXILIARY_TASKS],
                                     model,
                                     buffer,
                                     experiment_config)

    learning_algorithm = SACDAC(model=model,
                                policy_opt=policy_opt,
                                qs_opt=qs_opt,
                                alpha_opt=alpha_opt,
                                learn_alpha=experiment_config[c.LEARN_ALPHA],
                                buffer=buffer,
                                algo_params=experiment_config,
                                aux_tasks=aux_tasks)

    expert_buffer = make_buffer(experiment_config[c.BUFFER_SETTING], seed, experiment_config[c.EXPERT_BUFFER])
    discriminator = make_model(experiment_config[c.DISCRIMINATOR_SETTING])
    discriminator_opt = make_optimizer(discriminator.parameters(), experiment_config[c.OPTIMIZER_SETTING][c.DISCRIMINATOR])
    dac = DAC(discriminator=discriminator,
              discriminator_opt=discriminator_opt,
              expert_buffer=expert_buffer,
              learning_algorithm=learning_algorithm,
              algo_params=experiment_config)

    load_model = experiment_config.get(c.LOAD_MODEL, False)
    if load_model:
        learning_algorithm.load_state_dict(torch.load(load_model))

    agent = ACAgent(model=model,
                    learning_algorithm=dac,
                    preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    summary_writer, save_path = make_summary_writer(save_path=save_path,
                                                    algo=c.DAC,
                                                    cfg=experiment_config)
    evaluation_env = None
    evaluation_agent = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0):
        evaluation_env = make_env(experiment_config[c.ENV_SETTING], seed + 1)
        evaluation_agent = ACAgent(model=model,
                                   learning_algorithm=None,
                                   preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    train(agent=agent,
          evaluation_agent=evaluation_agent,
          train_env=train_env,
          evaluation_env=evaluation_env,
          buffer_preprocess=buffer_preprocessing,
          experiment_settings=experiment_config,
          auxiliary_reward=experiment_config[c.EVALUATION_REWARD_FUNC],
          summary_writer=summary_writer,
          save_path=save_path)
