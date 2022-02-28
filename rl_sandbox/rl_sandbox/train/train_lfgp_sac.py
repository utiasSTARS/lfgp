import gzip
import _pickle as pickle
import torch

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
from rl_sandbox.utils import make_summary_writer, set_seed


def train_lfgp_sac(experiment_config, return_agent_only=False, no_expert_buffers=False):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())

    set_seed(seed)
    if not return_agent_only:
        train_env = make_env(experiment_config[c.ENV_SETTING], seed)
    buffer = make_buffer(experiment_config[c.BUFFER_SETTING], seed, experiment_config[c.BUFFER_SETTING].get(c.LOAD_BUFFER, False))

    intentions = make_model(experiment_config[c.INTENTIONS_SETTING])
    policy_opt = make_optimizer(intentions.policy_parameters, experiment_config[c.OPTIMIZER_SETTING][c.INTENTIONS])
    qs_opt = make_optimizer(intentions.qs_parameters, experiment_config[c.OPTIMIZER_SETTING][c.QS])
    alpha_opt = make_optimizer([intentions.log_alpha], experiment_config[c.OPTIMIZER_SETTING][c.ALPHA])

    discriminator = make_model(experiment_config[c.DISCRIMINATOR_SETTING])

    load_transfer_exp_settings = experiment_config.get(c.LOAD_TRANSFER_EXP_SETTINGS, False)
    load_model = experiment_config.get(c.LOAD_MODEL, False)

    if load_transfer_exp_settings:
        from rl_sandbox.train.transfer import load_and_transfer
        old_config = load_and_transfer(load_transfer_exp_settings, load_model, intentions, buffer, 
                                       experiment_config, experiment_config[c.DEVICE].index, discriminator)
        load_model = False  # so we don't do learning algorithm load later

    aux_tasks = make_auxiliary_tasks(experiment_config[c.AUXILIARY_TASKS],
                                     intentions,
                                     buffer,
                                     experiment_config)

    sac_intentions = UpdateSACDACIntentions(model=intentions,
                                            policy_opt=policy_opt,
                                            qs_opt=qs_opt,
                                            alpha_opt=alpha_opt,
                                            learn_alpha=experiment_config[c.LEARN_ALPHA],
                                            buffer=buffer,
                                            algo_params=experiment_config,
                                            aux_tasks=aux_tasks)

    assert experiment_config[c.NUM_TASKS] == len(experiment_config[c.EXPERT_BUFFERS]) or \
           (c.HANDCRAFT_REWARDS in experiment_config[c.DISCRIMINATOR_SETTING][c.KWARGS].keys() and
           experiment_config[c.NUM_TASKS] == len(experiment_config[c.EXPERT_BUFFERS]) +
            len(experiment_config[c.DISCRIMINATOR_SETTING][c.KWARGS][c.HANDCRAFT_REWARDS]))
    expert_buffers = []
    if not no_expert_buffers:
        for load_path in experiment_config[c.EXPERT_BUFFERS]:
            # drop memory size for expert buffers to only what is needed
            with gzip.open(load_path, "rb") as f:
                data = pickle.load(f)
                experiment_config[c.BUFFER_SETTING][c.KWARGS][c.MEMORY_SIZE] = data[c.MEMORY_SIZE]

            expert_buffers.append(make_buffer(experiment_config[c.BUFFER_SETTING], seed, load_path))

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
        learning_algorithm.load_state_dict(torch.load(load_model))

    agent = SACXAgent(scheduler=scheduler,
                      intentions=intentions,
                      learning_algorithm=learning_algorithm,
                      scheduler_period=experiment_config[c.SCHEDULER_SETTING][c.TRAIN][c.SCHEDULER_PERIOD],
                      preprocess=experiment_config[c.EVALUATION_PREPROCESSING])
    evaluation_env = None
    evaluation_agent = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0) and not return_agent_only:
        evaluation_env = make_env(experiment_config[c.ENV_SETTING], seed + 1)
        evaluation_agent = SACXAgent(scheduler=make_model(experiment_config[c.SCHEDULER_SETTING][c.EVALUATION]),
                                     intentions=intentions,
                                     learning_algorithm=None,
                                     scheduler_period=experiment_config[c.SCHEDULER_SETTING][c.EVALUATION][c.SCHEDULER_PERIOD],
                                     preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    if not return_agent_only:
        summary_writer, save_path = make_summary_writer(save_path=save_path, algo=c.LFGP_NS if isinstance(scheduler, FixedScheduler) else c.LFGP, cfg=experiment_config)

    if load_transfer_exp_settings:
        from rl_sandbox.train.transfer import transfer_pretrain
        transfer_pretrain(learning_algorithm, experiment_config, old_config, update_intentions)

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
