import torch
import numpy as np
import functools
import timeit
from torch.multiprocessing import Pool

import rl_sandbox.constants as c

# from rl_sandbox.algorithms.sac_x.intentions_update.sac_intentions
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.buffers.ram_buffer import NumPyBuffer, NextStateNumPyBuffer
from rl_sandbox.buffers.torch_pin_buffer import TorchPinBuffer
from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import MultiTaskFullyConnectedSquashedGaussianSAC
from rl_sandbox.examples.eval_tools.utils import load_model


def load_and_transfer(load_transfer_exp_settings, load_model_file, intentions, buffer, experiment_config, device,
                      discriminator=None):
    if discriminator is not None:
        old_config, _, old_agent, old_discriminator = load_model(
            0, load_transfer_exp_settings, load_model_file, include_env=False, device=device)
    else:
        old_config, _, old_agent = load_model(0, load_transfer_exp_settings, load_model_file, include_env=False,
                                                device=device)
        old_discriminator = None
    transfer_existing_weights(old_agent.model, old_config, intentions, experiment_config, 
                              old_discriminator, discriminator)

    buffer = transfer_existing_buffer(buffer, old_config, experiment_config)
    return old_config


def transfer_pretrain(learning_algorithm, experiment_config, old_config, update_intentions):
    # workaround to start learning algorithm quickly
    learning_algorithm.step = experiment_config[c.BUFFER_WARMUP]
    learning_algorithm.update_intentions.step = experiment_config[c.BUFFER_WARMUP]

    _, _, _, _, unshared_new_i = get_shared_aux_for_transfer(old_config, experiment_config)

    if len(unshared_new_i) > 0:
        # all new rewards were in the old model, so no need to pretrain anything
        print("starting transfer pretrain")
        for i in range(experiment_config[c.TRANSFER_PRETRAIN]):
            update_info = {}
            tic = timeit.default_timer()
            updated_intentions, intentions_info = update_intentions.update(*[None] * 8, update_buffer=False)
            toc = timeit.default_timer()
            if updated_intentions:
                update_info[c.INTENTIONS_UPDATE_TIME] = toc - tic
                update_info.update(intentions_info)
        print("After transfer short retrain, Final pi loss: %.3f, final q1 loss %.3f" % (update_info['pi_loss'][0],
                                                                                            update_info['q1_loss'][0]))

def get_shared_aux_for_transfer(old_config, new_config):
    new_aux_r_list = new_config[c.AUXILIARY_REWARDS]._aux_rewards
    new_aux_r_set = set(new_aux_r_list)
    old_aux_r_list = old_config[c.AUXILIARY_REWARDS]._aux_rewards
    old_aux_r_set = set(old_aux_r_list)
    shared_aux_r = old_aux_r_set & new_aux_r_set
    unshared_new_aux_r = new_aux_r_set ^ shared_aux_r  # XOR gives exclusively new rewards

    ignore_aux_str_list = new_config.get(c.TRANSFER_AUX_IGNORE, None)
    if ignore_aux_str_list is not None:
        for r_func in new_aux_r_list:
            r_func_str = r_func.__qualname__
            if r_func_str in ignore_aux_str_list and r_func in shared_aux_r:
                shared_aux_r.remove(r_func)
                unshared_new_aux_r.add(r_func)

    shared_old_i = [i for i, e in enumerate(old_aux_r_list) if e in shared_aux_r]
    shared_new_i = [i for i, e in enumerate(new_aux_r_list) if e in shared_aux_r]
    unshared_new_i = [i for i, e in enumerate(new_aux_r_list) if e in unshared_new_aux_r]

    return shared_old_i, shared_new_i, old_aux_r_list, new_aux_r_list, unshared_new_i


def transfer_existing_weights(old_model, old_config, new_model, new_config,
                              old_discriminator=None, new_discriminator=None):
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

        if old_discriminator is not None:
            # fc layers are completely shared, regardless of number of output tasks
            new_discriminator.fc_layers.load_state_dict(old_discriminator.fc_layers.state_dict())

            new_discriminator.output.weight.data[shared_new_i] = old_discriminator.output.weight.data[shared_old_i]

    else:
        raise NotImplementedError()


def get_new_reward(rf, obs_size, obs_act):
    return rf(observation=obs_act[:obs_size], action=obs_act[obs_size:], info=None)


def transfer_existing_buffer(buffer, old_config, new_config):
    shared_old_i, shared_new_i, old_aux_r_list, new_aux_r_list, unshared_new_i\
        = get_shared_aux_for_transfer(old_config, new_config)

    if new_config[c.TRANSFER_BUFFER_DOWNSAMPLE] < 1 or new_config[c.TRANSFER_BUFFER_MAX_INDEX] is not None:
        if type(buffer) == TorchPinBuffer:
            buffer.downsample(new_config[c.TRANSFER_BUFFER_DOWNSAMPLE], new_config[c.TRANSFER_BUFFER_MAX_INDEX])
        elif type(buffer.buffer) == NumPyBuffer:
            new_config[c.BUFFER_SETTING][c.STORE_NEXT_OBSERVATION] = True  # convert to NextStateNumPyBuffer

            new_ns_buffer = buffer.buffer.get_next_state_buffer(
                new_config[c.TRANSFER_BUFFER_DOWNSAMPLE], new_config[c.TRANSFER_BUFFER_MAX_INDEX])
            new_buffer = make_buffer(new_config[c.BUFFER_SETTING], new_config[c.SEED], False)
            new_buffer.buffer = new_ns_buffer
            buffer = new_buffer

        elif type(buffer.buffer) == NextStateNumPyBuffer:
            buffer.buffer.downsample(new_config[c.TRANSFER_BUFFER_DOWNSAMPLE], new_config[c.TRANSFER_BUFFER_MAX_INDEX])

    if c.DISCRIMINATOR_SETTING in new_config:
        print("LfGP transfer, no reward loading needed.")
    else:
        if hasattr(buffer, 'buffer'):
            actual_buffer = buffer.buffer
        else:
            actual_buffer = buffer

        if len(unshared_new_i) == 0:
            # all new rewards were in the old model, so no need to get new reward data for existing observations
            actual_buffer.rewards = actual_buffer.rewards[:, shared_old_i]
        else:
            print('starting reward gen for new tasks')
            new_rewards = np.zeros([actual_buffer.rewards.shape[0], len(new_aux_r_list)])

            # do all this transition to np stuff because not sure that buffer on gpu would handle multiprocessing properly

            if hasattr(actual_buffer, 'device'):
                new_rewards[:, shared_new_i] = actual_buffer.rewards.cpu()
            else:
                new_rewards[:, shared_new_i] = actual_buffer.rewards
            new_r_funcs = tuple(np.array(new_config[c.AUXILIARY_REWARDS]._aux_rewards)[unshared_new_i])

            for rf, new_i in zip(new_r_funcs, unshared_new_i):
                obss, _, acts, rews, _, infos, _ = actual_buffer.get_transitions(slice(actual_buffer._count))

                if hasattr(actual_buffer, 'device'):
                    obss = obss.cpu()
                    acts = acts.cpu()

                with Pool(processes=12) as pool:
                    new_task_rewards = np.array(pool.map(
                        functools.partial(get_new_reward, rf, obss.shape[-1]), 
                        np.concatenate([obss.squeeze(), acts], -1)))
                new_rewards[:actual_buffer._count, new_i] = new_task_rewards

            if hasattr(actual_buffer, 'device'):
                actual_buffer.rewards = torch.tensor(new_rewards).to(actual_buffer.device)
            else:
                actual_buffer.rewards = new_rewards
            print('reward gen for new tasks complete')
    return buffer