import numpy as np
import timeit
import torch
import torch.nn as nn

import rl_sandbox.constants as c

from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask

class BC:
    def __init__(self, model, optimizer, expert_buffer, algo_params, aux_tasks=AuxiliaryTask()):
        """ Basic behavioral cloning algorithm that minimizes the negative log likelihood
        """

        self._optimizer = optimizer
        self.model = model
        self.buffer = expert_buffer
        self.algo_params = algo_params
        self.step = 0

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))
        self._opt_epochs = algo_params.get(
            c.OPT_EPOCHS, c.DEFAULT_BC_PARAMS[c.OPT_EPOCHS])
        self._opt_batch_size = algo_params.get(
            c.OPT_BATCH_SIZE, c.DEFAULT_BC_PARAMS[c.OPT_BATCH_SIZE])
        self._accum_num_grad = algo_params.get(
            c.ACCUM_NUM_GRAD, c.DEFAULT_BC_PARAMS[c.ACCUM_NUM_GRAD])
        self._aux_tasks = aux_tasks

        assert self._opt_batch_size % self._accum_num_grad == 0
        self._num_samples_per_accum = self._opt_batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_BC_PARAMS[c.MAX_GRAD_NORM])

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.OPTIMIZER] = self._optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self._optimizer.load_state_dict(state_dict[c.OPTIMIZER])

    def _compute_bc_loss(self, obss, h_states, acts, lengths):
        p_acts, _ = self.model.deterministic_act_lprob(obss, h_states)
        return torch.sum((p_acts - acts.to(self.device)) ** 2)

    def clone_policy(self, obss, h_states, acts, lengths, update_info):
        tic = timeit.default_timer()

        self._optimizer.zero_grad()
        total_bc_loss = 0.
        for accum_i in range(self._accum_num_grad):
            opt_idxes = range(accum_i * self._num_samples_per_accum,
                              (accum_i + 1) * self._num_samples_per_accum)
            bc_loss = self._compute_bc_loss(obss[opt_idxes],
                                            h_states[opt_idxes],
                                            acts[opt_idxes],
                                            lengths[opt_idxes])
            bc_loss /= self._opt_batch_size
            bc_loss.backward()
            total_bc_loss += bc_loss.detach().cpu()

        nn.utils.clip_grad_norm_(self.model.parameters(),
                                 self._max_grad_norm)
        self._optimizer.step()

        update_info[c.BC_LOSS].append(total_bc_loss.numpy())
        update_info[c.POLICY_UPDATE_TIME].append(timeit.default_timer() - tic)

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.step += 1
        update_info = {}

        update_info[c.BC_LOSS] = []
        update_info[c.SAMPLE_TIME] = []
        update_info[c.POLICY_UPDATE_TIME] = []
        for e in range(self._opt_epochs):
            tic = timeit.default_timer()
            obss, h_states, acts, _, _, _, lengths = self.buffer.sample(self._opt_batch_size)
            update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)
            self.clone_policy(obss,
                              h_states,
                              acts,
                              lengths,
                              update_info)

            if e % 5000 == 0:
                print("Grad step %d | T loss: %.4f | T time: %.5f" %
                        ((self.step - 1) * self._opt_epochs + e, update_info[c.BC_LOSS][-1],
                        update_info[c.POLICY_UPDATE_TIME][-1]))

        return True, update_info


class MultitaskBC:
    def __init__(self, model, optimizer, expert_buffers, algo_params, aux_tasks=AuxiliaryTask()):
        self._optimizer = optimizer
        self.model = model
        self.buffers = expert_buffers
        self.num_tasks = len(self.buffers)
        self.algo_params = algo_params
        self.step = 0

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))
        self._opt_epochs = algo_params.get(
            c.OPT_EPOCHS, c.DEFAULT_BC_PARAMS[c.OPT_EPOCHS])
        self._opt_batch_size = algo_params.get(
            c.OPT_BATCH_SIZE, c.DEFAULT_BC_PARAMS[c.OPT_BATCH_SIZE])
        self._accum_num_grad = algo_params.get(
            c.ACCUM_NUM_GRAD, c.DEFAULT_BC_PARAMS[c.ACCUM_NUM_GRAD])
        self.coefficients = algo_params.get(c.COEFFICIENTS, np.ones(self.num_tasks))
        self._aux_tasks = aux_tasks

        assert self._opt_batch_size % self._accum_num_grad == 0
        self._num_samples_per_accum = self._opt_batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_BC_PARAMS[c.MAX_GRAD_NORM])

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.OPTIMIZER] = self._optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self._optimizer.load_state_dict(state_dict[c.OPTIMIZER])

    def _compute_bc_loss(self, obss, h_states, acts, lengths, task_i):
        p_acts, _ = self.model.deterministic_act_lprob(obss, h_states)
        return torch.sum((p_acts[:, task_i] - acts.to(self.device)) ** 2)

    def clone_policy(self, update_info):
        self._optimizer.zero_grad()

        # Sample and compute loss from i'th task
        total_bc_loss = 0.
        tic = timeit.default_timer()
        for task_i, buffer_i in enumerate(self.buffers):
            obss, h_states, acts, _, _, _, lengths = buffer_i.sample(self._opt_batch_size)

            per_task_loss = 0.
            for accum_i in range(self._accum_num_grad):
                opt_idxes = range(accum_i * self._num_samples_per_accum,
                                (accum_i + 1) * self._num_samples_per_accum)
                bc_loss = self._compute_bc_loss(obss[opt_idxes],
                                                h_states[opt_idxes],
                                                acts[opt_idxes],
                                                lengths[opt_idxes],
                                                task_i)
                bc_loss /= self._opt_batch_size * self.coefficients[task_i]
                bc_loss.backward()
                per_task_loss += bc_loss.detach().cpu()
                total_bc_loss += bc_loss.detach().cpu()
            update_info[f"{c.BC_LOSS}-task_{task_i}"].append(per_task_loss.numpy())
        update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)
        nn.utils.clip_grad_norm_(self.model.parameters(),
                                self._max_grad_norm)
        self._optimizer.step()
        update_info[c.BC_LOSS].append(total_bc_loss.numpy())
        update_info[c.POLICY_UPDATE_TIME].append(timeit.default_timer() - tic)

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.step += 1
        update_info = {}
        update_info[c.BC_LOSS] = []
        update_info[c.SAMPLE_TIME] = []
        update_info[c.POLICY_UPDATE_TIME] = []
        for task_i in range(self.num_tasks):
            update_info[f"{c.BC_LOSS}-task_{task_i}"] = []
        for e in range(self._opt_epochs):
            self.clone_policy(update_info)
            if e % 5000 == 0:
                print("Grad step %d | T loss: %.4f | T time: %.5f" %
                        ((self.step - 1) * self._opt_epochs + e, update_info[c.BC_LOSS][-1],
                        update_info[c.POLICY_UPDATE_TIME][-1]))
        return True, update_info
