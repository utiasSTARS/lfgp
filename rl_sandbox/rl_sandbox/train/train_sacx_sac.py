import numpy as np
import functools
from torch.multiprocessing import Pool
import torch.multiprocessing as multiprocessing
import timeit

import torch

import rl_sandbox.constants as c

from rl_sandbox.agents.hrl_agents import SACXAgent
from rl_sandbox.auxiliary_tasks.utils import make_auxiliary_tasks
from rl_sandbox.algorithms.sac_x.intentions_update.sac_intentions import UpdateSACIntentions
from rl_sandbox.algorithms.sac_x.schedulers_update.q_scheduler import UpdateQScheduler
from rl_sandbox.algorithms.sac_x.sac_x import SACX
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.buffers.ram_buffer import NumPyBuffer, NextStateNumPyBuffer
from rl_sandbox.envs.utils import make_env
from rl_sandbox.learning_utils import train
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import MultiTaskFullyConnectedSquashedGaussianSAC
from rl_sandbox.transforms.general_transforms import Identity
from rl_sandbox.utils import make_summary_writer, set_seed
from rl_sandbox.examples.eval_tools.utils import load_model
from rl_sandbox.train.transfer import load_and_transfer, transfer_pretrain


def get_shared_aux_for_transfer(old_config, new_config):
    new_aux_r_list = new_config[c.AUXILIARY_REWARDS]._aux_rewards
    new_aux_r_set = set(new_aux_r_list)
    old_aux_r_list = old_config[c.AUXILIARY_REWARDS]._aux_rewards
    old_aux_r_set = set(old_aux_r_list)
    shared_aux_r = old_aux_r_set & new_aux_r_set
    unshared_new_aux_r = new_aux_r_set ^ shared_aux_r  # XOR gives exclusively new rewards

    shared_old_i = [i for i, e in enumerate(old_aux_r_list) if e in shared_aux_r]
    shared_new_i = [i for i, e in enumerate(new_aux_r_list) if e in shared_aux_r]
    unshared_new_i = [i for i, e in enumerate(new_aux_r_list) if e in unshared_new_aux_r]

    return shared_old_i, shared_new_i, old_aux_r_list, new_aux_r_list, unshared_new_i


def transfer_existing_weights(old_model, old_config, new_model, new_config):
    assert type(new_model) == type(old_model), "Models must be same type to be transferred"
    shared_old_i, shared_new_i, old_aux_r_list, new_aux_r_list, _ = get_shared_aux_for_transfer(old_config, new_config)

    if type(new_model) == MultiTaskFullyConnectedSquashedGaussianSAC:
        act_size = int(old_model._policy[2].weight.shape[0] / len(old_aux_r_list) / 2)
        old_model_hidden_size = int(old_model._policy[0].weight.shape[0] / len(old_aux_r_list))
        new_model_hidden_size = int(new_model._policy[0].weight.shape[0] / len(new_aux_r_list))
        assert old_model_hidden_size == new_model_hidden_size, "Old model hidden size %s doesn't match" \
                   "new model hidden size %s" % (old_model_hidden_size, new_model_hidden_size)
        h_size = new_model_hidden_size

        old_weight_i = np.array([range(i*h_size, (i+1)*h_size) for i in shared_old_i]).flatten()
        old_weight_act_i = np.array([range(i*act_size*2, (i+1)*act_size*2) for i in shared_old_i]).flatten()
        new_weight_i = np.array([range(i*h_size, (i+1)*h_size) for i in shared_new_i]).flatten()
        new_weight_act_i = np.array([range(i*act_size*2, (i+1)*act_size*2) for i in shared_new_i]).flatten()

        # shared network can be done with state_dict since size is exactly the same
        new_model._shared_network.load_state_dict(old_model._shared_network.state_dict())

        model_section_strings = ['_policy', '_q1', '_q2']
        for s in model_section_strings:
            old = getattr(old_model, s)
            new = getattr(new_model, s)

            new[0].weight.data[new_weight_i] = old[0].weight.data[old_weight_i]
            new[0].bias.data[new_weight_i] = old[0].bias.data[old_weight_i]

            if s == '_policy':
                new[2].weight.data[new_weight_act_i] = old[2].weight.data[old_weight_act_i]
                new[2].bias.data[new_weight_act_i] = old[2].bias.data[old_weight_act_i]
            elif '_q' in s:
                new[2].weight.data[shared_new_i] = old[2].weight.data[shared_old_i]
                new[2].bias.data[shared_new_i] = old[2].bias.data[shared_old_i]
    else:
        raise NotImplementedError()


def get_new_reward(rf, obs_size, obs_act):
    return rf(observation=obs_act[:obs_size], action=obs_act[obs_size:], info=None)


def transfer_existing_buffer(buffer, old_config, new_config):
    shared_old_i, shared_new_i, old_aux_r_list, new_aux_r_list, unshared_new_i\
        = get_shared_aux_for_transfer(old_config, new_config)

    if new_config[c.TRANSFER_BUFFER_DOWNSAMPLE] < 1:
        if type(buffer.buffer) == NumPyBuffer:
            new_config[c.BUFFER_SETTING][c.STORE_NEXT_OBSERVATION] = True  # convert to NextStateNumPyBuffer

            new_ns_buffer = buffer.buffer.get_next_state_buffer(
                new_config[c.TRANSFER_BUFFER_DOWNSAMPLE], new_config[c.TRANSFER_BUFFER_MAX_INDEX])
            new_buffer = make_buffer(new_config[c.BUFFER_SETTING], new_config[c.SEED], False)
            new_buffer.buffer = new_ns_buffer
            buffer = new_buffer

        elif type(buffer.buffer) == NextStateNumPyBuffer:
            buffer.buffer.downsample(new_config[c.TRANSFER_BUFFER_DOWNSAMPLE])

    if len(unshared_new_i) == 0:
        # all new rewards were in the old model, so no need to get new reward data for existing observations
        buffer.buffer.rewards = buffer.buffer.rewards[:, shared_old_i]
    else:
        print('starting reward gen for new tasks')
        new_rewards = np.zeros([buffer.buffer.rewards.shape[0], len(new_aux_r_list)])
        new_rewards[:, shared_new_i] = buffer.buffer.rewards
        new_r_funcs = tuple(np.array(new_config[c.AUXILIARY_REWARDS]._aux_rewards)[unshared_new_i])

        for rf, new_i in zip(new_r_funcs, unshared_new_i):
            obss, _, acts, rews, _, infos, _ = buffer.buffer.get_transitions(slice(buffer.buffer._count))
            with Pool(processes=12) as pool:
                new_task_rewards = np.array(pool.map(
                    functools.partial(get_new_reward, rf, obss.shape[-1]), np.concatenate([obss.squeeze(), acts], -1)))
            new_rewards[:buffer.buffer._count, new_i] = new_task_rewards

        buffer.buffer.rewards = new_rewards
        print('reward gen for new tasks complete')
    return buffer


def train_sacx_sac(experiment_config):
    seed = experiment_config[c.SEED]
    save_path = experiment_config.get(c.SAVE_PATH, None)
    buffer_preprocessing = experiment_config.get(c.BUFFER_PREPROCESSING, Identity())

    set_seed(seed)
    train_env = make_env(experiment_config[c.ENV_SETTING], seed)
    buffer = make_buffer(experiment_config[c.BUFFER_SETTING], seed, experiment_config[c.BUFFER_SETTING].get(c.LOAD_BUFFER, False))
    intentions = make_model(experiment_config[c.INTENTIONS_SETTING])

    policy_opt = make_optimizer(intentions.policy_parameters, experiment_config[c.OPTIMIZER_SETTING][c.INTENTIONS])
    qs_opt = make_optimizer(intentions.qs_parameters, experiment_config[c.OPTIMIZER_SETTING][c.QS])
    alpha_opt = make_optimizer([intentions.log_alpha], experiment_config[c.OPTIMIZER_SETTING][c.ALPHA])

    load_transfer_exp_settings = experiment_config.get(c.LOAD_TRANSFER_EXP_SETTINGS, False)
    load_model = experiment_config.get(c.LOAD_MODEL, False)
    
    if load_transfer_exp_settings:
        old_config = load_and_transfer(load_transfer_exp_settings, load_model, intentions, buffer, 
                                       experiment_config, experiment_config[c.DEVICE].index)
        load_model = False  # so we don't do learning algorithm load later

    aux_tasks = make_auxiliary_tasks(experiment_config[c.AUXILIARY_TASKS],
                                     intentions,
                                     buffer,
                                     experiment_config)

    update_intentions = UpdateSACIntentions(model=intentions,
                                            policy_opt=policy_opt,
                                            qs_opt=qs_opt,
                                            alpha_opt=alpha_opt,
                                            learn_alpha=experiment_config[c.LEARN_ALPHA],
                                            buffer=buffer,
                                            algo_params=experiment_config,
                                            aux_tasks=aux_tasks)

    scheduler = make_model(experiment_config[c.SCHEDULER_SETTING][c.TRAIN])
    update_scheduler = UpdateQScheduler(model=scheduler,
                                        algo_params=experiment_config)

    learning_algorithm = SACX(update_scheduler=update_scheduler,
                              update_intentions=update_intentions,
                              algo_params=experiment_config)

    if load_model:
        learning_algorithm.load_state_dict(torch.load(load_model_file,
                                                      map_location='cuda:' + str(experiment_config[c.DEVICE].index)))

    agent = SACXAgent(scheduler=scheduler,
                      intentions=intentions,
                      learning_algorithm=learning_algorithm,
                      scheduler_period=experiment_config[c.SCHEDULER_SETTING][c.TRAIN][c.SCHEDULER_PERIOD],
                      preprocess=experiment_config[c.EVALUATION_PREPROCESSING])
    evaluation_env = None
    evaluation_agent = None
    if experiment_config.get(c.EVALUATION_FREQUENCY, 0):
        evaluation_env = make_env(experiment_config[c.ENV_SETTING], seed + 1)
        evaluation_agent = SACXAgent(scheduler=make_model(experiment_config[c.SCHEDULER_SETTING][c.EVALUATION]),
                                     intentions=intentions,
                                     learning_algorithm=None,
                                     scheduler_period=experiment_config[c.SCHEDULER_SETTING][c.EVALUATION][c.SCHEDULER_PERIOD],
                                     preprocess=experiment_config[c.EVALUATION_PREPROCESSING])

    summary_writer, save_path = make_summary_writer(save_path=save_path, algo=c.SACX, cfg=experiment_config)

    if load_transfer_exp_settings:
        transfer_pretrain(learning_algorithm, experiment_config, old_config, update_intentions)

    train(agent=agent,
          evaluation_agent=evaluation_agent,
          train_env=train_env,
          evaluation_env=evaluation_env,
          buffer_preprocess=buffer_preprocessing,
          auxiliary_reward=experiment_config[c.AUXILIARY_REWARDS].reward,
          experiment_settings=experiment_config,
          summary_writer=summary_writer,
          save_path=save_path)
