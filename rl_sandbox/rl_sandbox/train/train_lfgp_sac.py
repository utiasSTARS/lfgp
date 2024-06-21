import gzip
import _pickle as pickle
import torch
import json
import os
import glob

import rl_sandbox.constants as c


from rl_sandbox.agents.hrl_agents import SACXAgent
from rl_sandbox.auxiliary_tasks.utils import make_auxiliary_tasks
from rl_sandbox.algorithms.sac_x.intentions_update.dac_intentions import UpdateDACIntentions
from rl_sandbox.algorithms.sac_x.intentions_update.sac_intentions import UpdateSACDACIntentions
from rl_sandbox.algorithms.sac_x.schedulers import FixedScheduler
from rl_sandbox.algorithms.sac_x.schedulers_update.q_scheduler import UpdateDACQScheduler
from rl_sandbox.algorithms.sac_x.sac_x import SACX
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import train
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.transforms.general_transforms import Identity
from rl_sandbox.utils import make_summary_writer, set_seed, set_rng_state, check_load_latest_checkpoint, check_load_as_jumpoff_point
from rl_sandbox.envs.wrappers.frame_stack import FrameStackWrapper
from rl_sandbox.auxiliary_rewards.generic import FromEnvAuxiliaryReward


def train_lfgp_sac(experiment_config, return_agent_only=False, no_expert_buffers=False):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())

    save_path, add_time_tag_to_save_path = check_load_latest_checkpoint(experiment_config, save_path)
    save_path, add_time_tag_to_save_path = check_load_as_jumpoff_point(experiment_config, save_path, add_time_tag_to_save_path)
    buffer_end_idx = None
    if experiment_config.get(c.LOAD_BUFFER_START_INDEX, -1) >= 0:
        buffer_end_idx = experiment_config[c.LOAD_BUFFER_START_INDEX]

    set_seed(seed)
    if not return_agent_only:
        train_env = make_env(experiment_config[c.ENV_SETTING], seed)
    buffer = make_buffer(experiment_config[c.BUFFER_SETTING], seed, experiment_config[c.BUFFER_SETTING].get(c.LOAD_BUFFER, False),
                         end_idx=buffer_end_idx)

    intentions = make_model(experiment_config[c.INTENTIONS_SETTING])
    policy_opt = make_optimizer(intentions.policy_parameters, experiment_config[c.OPTIMIZER_SETTING][c.INTENTIONS])
    qs_opt = make_optimizer(intentions.qs_parameters, experiment_config[c.OPTIMIZER_SETTING][c.QS])
    alpha_opt = make_optimizer([intentions.log_alpha], experiment_config[c.OPTIMIZER_SETTING][c.ALPHA])

    discriminator = make_model(experiment_config[c.DISCRIMINATOR_SETTING])

    load_transfer_exp_settings = experiment_config.get(c.LOAD_TRANSFER_EXP_SETTINGS, False)
    load_model = experiment_config.get(c.LOAD_MODEL, False)

    if load_transfer_exp_settings and load_model:
        from rl_sandbox.train.transfer import load_and_transfer
        old_config = load_and_transfer(load_transfer_exp_settings, load_model, intentions, buffer,
                                       experiment_config, experiment_config[c.DEVICE].index, discriminator)
        load_model = False  # so we don't do learning algorithm load later

    # koopman aux tasks, not anything to do with lfgp
    aux_tasks = make_auxiliary_tasks(experiment_config[c.AUXILIARY_TASKS],
                                     intentions,
                                     buffer,
                                     experiment_config)

    assert experiment_config[c.NUM_TASKS] == len(experiment_config[c.EXPERT_BUFFERS]) or \
           (c.HANDCRAFT_REWARDS in experiment_config[c.DISCRIMINATOR_SETTING][c.KWARGS].keys() and
           experiment_config[c.NUM_TASKS] == len(experiment_config[c.EXPERT_BUFFERS]) +
            len(experiment_config[c.DISCRIMINATOR_SETTING][c.KWARGS][c.HANDCRAFT_REWARDS]))

    frame_stack = 1
    for wrap_dict in experiment_config[c.ENV_SETTING][c.ENV_WRAPPERS]:
        if wrap_dict[c.WRAPPER] == FrameStackWrapper:
            frame_stack = wrap_dict[c.KWARGS][c.NUM_FRAMES]

    expert_buffers = []

    # handle old code with old expert buffer options
    expert_amounts = experiment_config.get(c.EXPERT_AMOUNTS, [None] * experiment_config[c.NUM_TASKS])
    if not no_expert_buffers:
        expert_buffer_settings = experiment_config.get(c.EXPERT_BUFFER_SETTING, experiment_config[c.BUFFER_SETTING])

    if not no_expert_buffers:
        for load_path, amount in zip(experiment_config[c.EXPERT_BUFFERS], expert_amounts):
            expert_buffers.append(make_buffer(expert_buffer_settings, seed, load_path, end_idx=amount,
                                              match_load_size=True, frame_stack_load=frame_stack))

    if c.FT_EXPERT_BUFFERS in experiment_config:
        ft_expert_buffers = []
        for load_path, amount in zip(experiment_config[c.FT_EXPERT_BUFFERS], expert_amounts):
            ft_expert_buffers.append(make_buffer(expert_buffer_settings, seed, load_path, end_idx=amount,
                                                 match_load_size=True, frame_stack_load=frame_stack))
        for buf_i in range(len(expert_buffers)):
            expert_buffers[buf_i].merge(ft_expert_buffers[buf_i])

    sac_intentions = UpdateSACDACIntentions(model=intentions,
                                            policy_opt=policy_opt,
                                            qs_opt=qs_opt,
                                            alpha_opt=alpha_opt,
                                            learn_alpha=experiment_config[c.LEARN_ALPHA],
                                            buffer=buffer,
                                            algo_params=experiment_config,
                                            aux_tasks=aux_tasks,
                                            expert_buffers=expert_buffers)

    discriminator_opt = make_optimizer(discriminator.parameters(), experiment_config[c.OPTIMIZER_SETTING][c.DISCRIMINATOR])
    update_intentions = UpdateDACIntentions(discriminator=discriminator,
                                            discriminator_opt=discriminator_opt,
                                            expert_buffers=expert_buffers,
                                            learning_algorithm=sac_intentions,
                                            algo_params=experiment_config)

    scheduler = make_model(experiment_config[c.SCHEDULER_SETTING][c.TRAIN])
    update_scheduler = UpdateDACQScheduler(model=scheduler,
                                           reward_function=discriminator,
                                           algo_params=experiment_config)

    learning_algorithm = SACX(update_scheduler=update_scheduler,
                              update_intentions=update_intentions,
                              algo_params=experiment_config)

    if load_model:
        state_dict = torch.load(load_model, map_location=experiment_config[c.DEVICE])
        learning_algorithm.load_state_dict(state_dict)
        set_rng_state(state_dict[c.TORCH_RNG_STATE], state_dict[c.NP_RNG_STATE])

    agent = SACXAgent(scheduler=scheduler,
                      intentions=intentions,
                      learning_algorithm=learning_algorithm,
                      scheduler_period=experiment_config[c.SCHEDULER_SETTING][c.TRAIN][c.SCHEDULER_PERIOD],
                      preprocess=experiment_config[c.EVALUATION_PREPROCESSING])
    evaluation_env = None
    evaluation_agent = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0) and not return_agent_only:
        if experiment_config[c.ENV_SETTING][c.ENV_TYPE] == c.PANDA_RL_ENVS:
            evaluation_env = train_env
        else:
            evaluation_env = make_env(experiment_config[c.ENV_SETTING], seed + 1)
        evaluation_agent = SACXAgent(scheduler=make_model(experiment_config[c.SCHEDULER_SETTING][c.EVALUATION]),
                                     intentions=intentions,
                                     learning_algorithm=None,
                                     scheduler_period=experiment_config[c.SCHEDULER_SETTING][c.EVALUATION][c.SCHEDULER_PERIOD],
                                     preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    if not return_agent_only:
        summary_writer, save_path = make_summary_writer(
            save_path=save_path, algo=c.LFGP_NS if isinstance(scheduler, FixedScheduler) else c.LFGP,
            cfg=experiment_config, add_time_tag=add_time_tag_to_save_path)

    if load_transfer_exp_settings:
        if not load_model:
            from rl_sandbox.train.transfer import load_settings
            old_config = load_settings(load_transfer_exp_settings)
        from rl_sandbox.train.transfer import transfer_pretrain
        import ipdb; ipdb.set_trace()
        transfer_pretrain(learning_algorithm, experiment_config, old_config, update_intentions)

    if not hasattr(experiment_config[c.AUXILIARY_REWARDS], 'reward') and not return_agent_only:
        aux_reward = FromEnvAuxiliaryReward(train_env, experiment_config[c.AUXILIARY_REWARDS])
        experiment_config[c.AUXILIARY_REWARDS] = aux_reward

    if return_agent_only:
        return agent
    else:
        train(agent=agent,
              evaluation_agent=evaluation_agent,
              train_env=train_env,
              evaluation_env=evaluation_env,
              buffer_preprocess=buffer_preprocessing,
              auxiliary_reward=experiment_config[c.AUXILIARY_REWARDS].reward,
              experiment_settings=experiment_config,
              summary_writer=summary_writer,
              save_path=save_path)
