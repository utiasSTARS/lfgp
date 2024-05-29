import os
import glob

import torch

import rl_sandbox.constants as c

from rl_sandbox.algorithms.dac.sac import SACDAC
from rl_sandbox.algorithms.dac.dac import DAC
from rl_sandbox.auxiliary_tasks.utils import make_auxiliary_tasks
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import train
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.agents.rl_agents import ACAgent, ACAgentEUniformExplorer
from rl_sandbox.transforms.general_transforms import Identity
from rl_sandbox.utils import make_summary_writer, set_seed, set_rng_state, check_load_latest_checkpoint
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper

def train_dac_sac(experiment_config):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())

    save_path, add_time_tag_to_save_path = check_load_latest_checkpoint(experiment_config, save_path)

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

    frame_stack = 1
    for wrap_dict in experiment_config[c.ENV_SETTING][c.ENV_WRAPPERS]:
        if wrap_dict[c.WRAPPER] == FrameStackWrapper:
            frame_stack = wrap_dict[c.KWARGS][c.NUM_FRAMES]

    # handle old code without expert amount option
    expert_amount = experiment_config.get(c.EXPERT_AMOUNT, None)
    expert_buffer_settings = experiment_config.get(c.EXPERT_BUFFER_SETTING, experiment_config[c.BUFFER_SETTING])

    expert_buffer = make_buffer(expert_buffer_settings, seed, experiment_config[c.EXPERT_BUFFER],
                                end_idx=expert_amount, match_load_size=True, frame_stack_load=frame_stack)

    if c.FT_EXPERT_BUFFER in experiment_config:
        ft_expert_buffer = make_buffer(expert_buffer_settings, seed, experiment_config[c.FT_EXPERT_BUFFER],
                                       end_idx=expert_amount, match_load_size=True, frame_stack_load=frame_stack)
        expert_buffer.merge(ft_expert_buffer)

    learning_algorithm = SACDAC(model=model,
                                policy_opt=policy_opt,
                                qs_opt=qs_opt,
                                alpha_opt=alpha_opt,
                                learn_alpha=experiment_config[c.LEARN_ALPHA],
                                buffer=buffer,
                                algo_params=experiment_config,
                                aux_tasks=aux_tasks,
                                expert_buffer=expert_buffer)

    discriminator = make_model(experiment_config[c.DISCRIMINATOR_SETTING])
    discriminator_opt = make_optimizer(discriminator.parameters(), experiment_config[c.OPTIMIZER_SETTING][c.DISCRIMINATOR])
    dac = DAC(discriminator=discriminator,
              discriminator_opt=discriminator_opt,
              expert_buffer=expert_buffer,
              learning_algorithm=learning_algorithm,
              algo_params=experiment_config)

    load_model = experiment_config.get(c.LOAD_MODEL, False)
    if load_model:
        state_dict = torch.load(load_model, map_location=experiment_config[c.DEVICE])
        dac.load_state_dict(state_dict)
        set_rng_state(state_dict[c.TORCH_RNG_STATE], state_dict[c.NP_RNG_STATE])

    # TODO add this as a proper option
    # agent = ACAgentEUniformExplorer(model=model, learning_algorithm=dac,
    #                                 prob_explore_ep=.2, prob_explore_act=.05, max_repeat=41, min_repeat=40)

    agent = ACAgent(model=model,
                    learning_algorithm=dac,
                    preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    # overwrites the save path with a time tag
    summary_writer, save_path = make_summary_writer(save_path=save_path,
                                                    algo=c.DAC,
                                                    cfg=experiment_config,
                                                    add_time_tag=add_time_tag_to_save_path)
    evaluation_env = None
    evaluation_agent = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0):
        if experiment_config[c.ENV_SETTING][c.ENV_TYPE] == c.PANDA_RL_ENVS:
            evaluation_env = train_env
        else:
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
