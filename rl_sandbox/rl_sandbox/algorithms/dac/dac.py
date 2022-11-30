import numpy as np
import timeit
import torch
import torch.autograd as autograd
import torch.nn as nn

import rl_sandbox.constants as c


class DAC:
    def __init__(self, discriminator, discriminator_opt, expert_buffer, learning_algorithm, algo_params):
        """ DAC Algorithm: https://arxiv.org/abs/1809.02925
        """
        self.learning_algorithm = learning_algorithm

        self.expert_buffer = expert_buffer
        self.buffer = self.learning_algorithm.buffer

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self.discriminator = discriminator
        self.discriminator_opt = discriminator_opt
        self._discriminator_batch_size = algo_params[c.DISCRIMINATOR_BATCH_SIZE]
        self._gp_lambda = algo_params[c.GRADIENT_PENALTY_LAMBDA]
        self.device = algo_params[c.DEVICE]
        self.algo_params = algo_params

        self._expbuf_last_sample_prop = algo_params.get(c.DISCRIMINATOR_EXPBUF_LAST_SAMPLE_PROP, 0)
        if self._expbuf_last_sample_prop > 0:
            # get final transition indices of each buffer
            if type(self.expert_buffer.observations) == torch.Tensor:
                obs = self.expert_buffer.observations[:len(self.expert_buffer)].cpu().numpy()
                next_obs = self.expert_buffer.next_observations[:len(self.expert_buffer)].cpu().numpy()
            else:
                obs = self.expert_buffer.observations[:len(self.expert_buffer)]
                next_obs = self.expert_buffer.next_observations[:len(self.expert_buffer)]

            self._end_indices = np.argwhere(np.invert(np.all(obs[1:] == next_obs[:-1], axis=1))).flatten()
            others = np.argwhere(np.all(obs[1:] == next_obs[:-1], axis=1)).flatten()

            # this might be an end, but adding to others anyways
            self._other_indices = np.concatenate([others, [len(self.expert_buffer) - 1]])

    def state_dict(self):
        state_dict = dict()
        state_dict[c.ALGORITHM] = self.learning_algorithm.state_dict()
        state_dict[c.DISCRIMINATOR] = self.discriminator.state_dict()
        state_dict[c.DISCRIMINATOR_OPTIMIZER] = self.discriminator_opt.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.learning_algorithm.load_state_dict(state_dict[c.ALGORITHM])
        self.discriminator.load_state_dict(state_dict[c.DISCRIMINATOR])
        self.discriminator_opt.load_state_dict(state_dict[c.DISCRIMINATOR_OPTIMIZER])

    def update_discriminator(self):
        update_info = {}
        update_info[c.GAN_LOSS] = []
        update_info[c.GP_LOSS] = []
        update_info[c.DISCRIMINATOR_SAMPLE_TIME] = []
        update_info[c.DISCRIMINATOR_UPDATE_TIME] = []

        self.discriminator_opt.zero_grad()

        tic = timeit.default_timer()

        if self._expbuf_last_sample_prop > 0:
            num_ends = int(self._expbuf_last_sample_prop * self._discriminator_batch_size)
            num_others = self._discriminator_batch_size - num_ends
            indices = []
            other_indices_samp = torch.multinomial(
                torch.ones(len(self._other_indices)), num_others, replacement=True)
            indices.extend(self._other_indices[other_indices_samp])
            end_indices_samp = torch.multinomial(
                torch.ones(len(self._end_indices)), num_ends, replacement=True)
            indices.extend(self._end_indices[end_indices_samp])
            obss_e, _, acts_e, _, _, _, lengths_e = self.expert_buffer.sample(
                self._discriminator_batch_size, idxes=indices)
        else:
            obss_e, _, acts_e, _, _, _, lengths_e = self.expert_buffer.sample(self._discriminator_batch_size)
        idxes = lengths_e.unsqueeze(-1).repeat(1, *obss_e.shape[2:]).unsqueeze(1)
        obss_e = torch.gather(obss_e, axis=1, index=idxes - 1)[:, 0, :]
        absorbing_idx_e = torch.where(obss_e[:, -1])[0]
        acts_e[absorbing_idx_e] = 0
        obss_e = self.train_preprocessing(obss_e)

        obss, _, acts, _, _, _, lengths = self.buffer.sample(self._discriminator_batch_size)
        idxes = lengths.unsqueeze(-1).repeat(1, *obss.shape[2:]).unsqueeze(1)
        obss = torch.gather(obss, axis=1, index=idxes - 1)[:, 0, :]
        absorbing_idx = torch.where(obss[:, -1])[0]
        acts[absorbing_idx] = 0
        obss = self.train_preprocessing(obss)

        update_info[c.DISCRIMINATOR_SAMPLE_TIME].append(timeit.default_timer() - tic)

        tic = timeit.default_timer()

        # GAIL Loss
        logits = self.discriminator(obss, acts)
        logits_e = self.discriminator(obss_e, acts_e)

        zeros = torch.zeros(self._discriminator_batch_size,
                            dtype=torch.float,
                            device=self.device)

        ones = torch.ones(self._discriminator_batch_size,
                          dtype=torch.float,
                          device=self.device)
        gan_loss = (torch.nn.BCEWithLogitsLoss()(logits[:, 0], zeros) + \
            torch.nn.BCEWithLogitsLoss()(logits_e[:, 0], ones)) / 2
        gan_loss.backward()

        # Gradient Penalty Loss
        obss_noise = torch.distributions.Uniform(
            torch.tensor(0., device=obss.device), torch.tensor(1., device=obss.device)).sample(obss.shape)
        acts_noise = torch.distributions.Uniform(
            torch.tensor(0., device=obss.device), torch.tensor(1., device=obss.device)).sample(acts.shape)
        obss_mixture = obss_noise * obss + (1 - obss_noise) * obss_e
        acts_mixture = acts_noise * acts + (1 - acts_noise) * acts_e
        obss_mixture.requires_grad_(True)
        acts_mixture.requires_grad_(True)

        output_mixture = self.discriminator(obss_mixture, acts_mixture)

        gradients = torch.cat(autograd.grad(outputs=output_mixture,
                                            inputs=(obss_mixture, acts_mixture),
                                            grad_outputs=torch.ones(output_mixture.size(), device=self.device),
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True), dim=-1)

        gp_loss = self._gp_lambda * torch.mean((torch.norm(gradients, dim=-1) - 1) ** 2)
        gp_loss.backward()

        self.discriminator_opt.step()

        update_info[c.DISCRIMINATOR_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.GAN_LOSS].append(gan_loss.cpu().detach().numpy())
        update_info[c.GP_LOSS].append(gp_loss.cpu().detach().numpy())

        return update_info


    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        # NOTE: The states are already processed such that the absorbing state padding is done through envs/wrappers/absorbing_state.py
        if curr_obs[:, -1] == 1:
            act[:] = 0
        self.buffer.push(obs=curr_obs,
                         h_state=curr_h_state,
                         act=act,
                         rew=0.,
                         done=[False],
                         info=info,
                         next_obs=next_obs,
                         next_h_state=next_h_state)

        # The reward will be computed in the underlying policy learning algorithm
        # NOTE: Disable _store_to_buffer in learning algorithm
        discriminator_update_info = {}
        if self.learning_algorithm.step >= self.learning_algorithm._buffer_warmup:
            discriminator_update_info  = self.update_discriminator()
        updated, update_info = self.learning_algorithm.update(self.discriminator, next_obs, next_h_state)

        update_info.update(discriminator_update_info)

        return updated, update_info
