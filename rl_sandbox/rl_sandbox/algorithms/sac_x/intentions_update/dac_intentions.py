from pydoc import replace
import numpy as np
import timeit
import torch
import torch.autograd as autograd
import torch.nn as nn

from rl_sandbox.envs.wrappers.absorbing_state import check_absorbing
import rl_sandbox.constants as c


class UpdateDACIntentions:
    def __init__(self, discriminator, discriminator_opt, expert_buffers, learning_algorithm, algo_params):
        """ DAC Algorithm: https://arxiv.org/abs/1809.02925
        This wraps around a learning algorithm.
        """
        self.learning_algorithm = learning_algorithm

        self.expert_buffers = expert_buffers
        self.buffer = self.learning_algorithm.buffer

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self.discriminator = discriminator
        self.discriminator_opt = discriminator_opt
        self._discriminator_batch_size = algo_params[c.DISCRIMINATOR_BATCH_SIZE]
        self._gp_lambda = algo_params[c.GRADIENT_PENALTY_LAMBDA]
        self.device = algo_params[c.DEVICE]
        self.algo_params = algo_params

        self._use_absorbing_state = check_absorbing(self.algo_params)

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))
        self._num_tasks = algo_params.get(c.NUM_TASKS, 1)
        self._action_dim = algo_params[c.ACTION_DIM]
        self._task_batch_size = self._discriminator_batch_size * self.discriminator._output_dim
        self._num_discrim_updates = algo_params.get(c.DISCRIMINATOR_NUM_UPDATES, 1)

        self._expbuf_last_sample_prop = algo_params.get(c.DISCRIMINATOR_EXPBUF_LAST_SAMPLE_PROP, 0)
        if self._expbuf_last_sample_prop > 0:
            self._end_indices = []
            self._other_indices = []
            # get final transition indices of each buffer
            for eb in self.expert_buffers:
                if type(eb.observations) == torch.Tensor:
                    obs = eb.observations[:len(eb)].cpu().numpy()
                    next_obs = eb.next_observations[:len(eb)].cpu().numpy()
                else:
                    obs = eb.observations[:len(eb)]
                    next_obs = eb.next_observations[:len(eb)]

                ends = np.argwhere(np.invert(np.all(obs[1:] == next_obs[:-1], axis=1))).flatten()
                others = np.argwhere(np.all(obs[1:] == next_obs[:-1], axis=1)).flatten()
                others = np.concatenate([others, [len(eb) - 1]])  # this might be an end, but adding to others anyways
                self._end_indices.append(ends)
                self._other_indices.append(others)

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

        # Need to gather the expert data from each buffer
        obss_e = []
        acts_e = []

        for i, expert_buffer in enumerate(self.expert_buffers):
            if self._expbuf_last_sample_prop > 0:
                num_ends = int(self._expbuf_last_sample_prop * self._discriminator_batch_size)
                num_others = self._discriminator_batch_size - num_ends
                indices = []
                other_indices_samp = torch.multinomial(
                    torch.ones(len(self._other_indices[i])), num_others, replacement=True)
                indices.extend(self._other_indices[i][other_indices_samp])
                end_indices_samp = torch.multinomial(
                    torch.ones(len(self._end_indices[i])), num_ends, replacement=True)
                indices.extend(self._end_indices[i][end_indices_samp])
                curr_obss_e, _, curr_acts_e, _, _, _, lengths = expert_buffer.sample(
                    self._discriminator_batch_size, idxes=indices)
            else:
                curr_obss_e, _, curr_acts_e, _, _, _, lengths = expert_buffer.sample(self._discriminator_batch_size)
            idxes = lengths.unsqueeze(-1).repeat(1, *curr_obss_e.shape[2:]).unsqueeze(1)
            curr_obss_e = torch.gather(curr_obss_e, axis=1, index=idxes - 1)[:, 0, :]
            obss_e.append(curr_obss_e)
            acts_e.append(curr_acts_e)

        obss_e = torch.cat(obss_e, dim=0)
        acts_e = torch.cat(acts_e, dim=0)

        if self._use_absorbing_state:
            absorbing_idx_e = torch.where(obss_e[:, -1])[0]
            acts_e[absorbing_idx_e] = 0
        obss_e = self.train_preprocessing(obss_e)

        obss, _, acts, _, _, _, lengths = self.buffer.sample(self._discriminator_batch_size)
        idxes = lengths.unsqueeze(-1).repeat(1, *obss.shape[2:]).unsqueeze(1)
        obss = torch.gather(obss, axis=1, index=idxes - 1)[:, 0, :]

        if self._use_absorbing_state:
            absorbing_idx = torch.where(obss[:, -1])[0]
            acts[absorbing_idx] = 0
        obss = self.train_preprocessing(obss)

        update_info[c.DISCRIMINATOR_SAMPLE_TIME].append(timeit.default_timer() - tic)

        tic = timeit.default_timer()

        # GAIL Loss
        logits = self.discriminator(obss, acts)

        if hasattr(self.discriminator, "_handcraft_rewards"):
            num_trainable_tasks = self._num_tasks - len(self.discriminator._handcraft_rewards)
            logits = logits[:, self.discriminator._trainable_indices]

            # For expert data, only take the corresponding task as reward
            tasks = torch.arange(num_trainable_tasks, device=self.device).repeat(
                self._discriminator_batch_size, 1).transpose(0, 1).flatten().unsqueeze(-1)

            logits_e = torch.gather(
                self.discriminator(obss_e, acts_e)[:, self.discriminator._trainable_indices], axis=1, index=tasks)
        else:
            num_trainable_tasks = self._num_tasks

            # For expert data, only take the corresponding task as reward
            tasks = torch.arange(num_trainable_tasks, device=self.device).repeat(
                self._discriminator_batch_size, 1).transpose(0, 1).flatten().unsqueeze(-1)

            logits_e = torch.gather(self.discriminator(obss_e, acts_e), axis=1, index=tasks)

        zeros = torch.zeros(self._task_batch_size,
                            dtype=torch.float,
                            device=self.device)

        ones = torch.ones(self._task_batch_size,
                          dtype=torch.float,
                          device=self.device)

        gan_loss = (torch.nn.BCEWithLogitsLoss()(logits.reshape(-1), zeros) + \
            torch.nn.BCEWithLogitsLoss()(logits_e.reshape(-1), ones)) / 2
        gan_loss.backward()

        # Gradient Penalty Loss
        obss = obss.repeat(num_trainable_tasks, 1)
        acts = acts.repeat(num_trainable_tasks, 1)
        obss_noise = torch.distributions.Uniform(
            torch.tensor(0., device=obss.device), torch.tensor(1., device=obss.device)).sample(obss.shape)
        acts_noise = torch.distributions.Uniform(
            torch.tensor(0., device=obss.device), torch.tensor(1., device=obss.device)).sample(acts.shape)

        obss_mixture = obss_noise * obss + (1 - obss_noise) * obss_e
        acts_mixture = acts_noise * acts + (1 - acts_noise) * acts_e
        obss_mixture.requires_grad_(True)
        acts_mixture.requires_grad_(True)

        output_mixture = self.discriminator(obss_mixture, acts_mixture)

        if self.discriminator._obs_only:
            grad_inputs = obss_mixture
        else:
            grad_inputs = (obss_mixture, acts_mixture)

        gradients = torch.cat(autograd.grad(outputs=output_mixture,
                                            inputs=grad_inputs,
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


    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state, update_buffer=True):
        if update_buffer:
            if curr_obs[:, -1] == 1 and self._use_absorbing_state:
                act[:] = 0
            self.buffer.push(obs=curr_obs,
                            h_state=curr_h_state,
                            act=act,
                            rew=rew if self.learning_algorithm._reward_model == c.SPARSE else 0.,
                            done=done,
                            info=info,
                            next_obs=next_obs,
                            next_h_state=next_h_state)

        # The reward will be computed in the underlying policy learning algorithm
        discriminator_update_info = {}
        if self.learning_algorithm.step >= self.learning_algorithm._buffer_warmup and \
                self.algo_params.get(c.REWARD_MODEL, "discriminator") == "discriminator":
            for i in range(self._num_discrim_updates):
                discriminator_update_info = self.update_discriminator()
        updated, update_info = self.learning_algorithm.update(self.discriminator, next_obs, next_h_state)

        update_info.update(discriminator_update_info)

        return updated, update_info


class UpdateDACIntentionsPlusHandcraft(UpdateDACIntentions):
    def __init__(self, discriminator, discriminator_opt, expert_buffers, learning_algorithm, algo_params):
        super().__init__(discriminator, discriminator_opt, expert_buffers, learning_algorithm, algo_params)
        self._num_tasks = algo_params.get(c.NUM_TRAIN_TASKS, 1)
        self._task_batch_size = self._discriminator_batch_size * self._num_tasks