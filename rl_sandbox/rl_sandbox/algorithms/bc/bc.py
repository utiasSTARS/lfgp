import copy
import numpy as np
import timeit
import torch
import torch.nn as nn

from torch.utils.data import BatchSampler, SubsetRandomSampler

import rl_sandbox.constants as c

from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask

class BC:
    def __init__(self, model, optimizer, expert_buffer, algo_params, aux_tasks=AuxiliaryTask()):
        """ Basic behavioral cloning algorithm that minimizes the negative log likelihood
        """

        self._optimizer = optimizer
        self.model = model
        self.expert_buffer = expert_buffer
        self.algo_params = algo_params
        self.step = 0

        self.device = algo_params.get(c.DEVICE, torch.device(c.CPU))
        self._opt_epochs = algo_params.get(
            c.OPT_EPOCHS, c.DEFAULT_BC_PARAMS[c.OPT_EPOCHS])
        self._opt_batch_size = algo_params.get(
            c.OPT_BATCH_SIZE, c.DEFAULT_BC_PARAMS[c.OPT_BATCH_SIZE])
        self._accum_num_grad = algo_params.get(
            c.ACCUM_NUM_GRAD, c.DEFAULT_BC_PARAMS[c.ACCUM_NUM_GRAD])
        self._overfit_tolerance = algo_params.get(
            c.OVERFIT_TOLERANCE, c.DEFAULT_BC_PARAMS[c.OVERFIT_TOLERANCE])
        self._aux_tasks = aux_tasks

        assert self._opt_batch_size % self._accum_num_grad == 0
        self._num_samples_per_accum = self._opt_batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_BC_PARAMS[c.MAX_GRAD_NORM])

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self._train_val_ratio = algo_params.get(
            c.VALIDATION_RATIO, c.DEFAULT_BC_PARAMS[c.VALIDATION_RATIO])
        num_val = int(len(self.expert_buffer) * self._train_val_ratio)
        num_train = len(self.expert_buffer) - num_val
        idxes = np.random.permutation(np.arange(len(self.expert_buffer)))
        self._val_sampler = BatchSampler(sampler=SubsetRandomSampler(idxes[num_train:]),
                                         batch_size=self._opt_batch_size,
                                         drop_last=True)
        self._train_sampler = BatchSampler(sampler=SubsetRandomSampler(idxes[:num_train]),
                                           batch_size=self._opt_batch_size,
                                           drop_last=True)

        self.best_validation_loss = np.inf
        self._overfit_count = 0
        self.overfitted = False
        self._curr_best_model = copy.deepcopy(self.model.state_dict())

    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.OPTIMIZER] = self._optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self._optimizer.load_state_dict(state_dict[c.OPTIMIZER])

    def _compute_bc_loss(self, obss, h_states, acts, lengths):
        # old method, not quite correct
        # dist, _, _ = self.model(obss, h_states, lengths=lengths)
        # return torch.sum((dist.mean - acts.to(self.device)) ** 2)

        p_acts, _ = self.model.deterministic_act_lprob(obss, h_states)
        return torch.sum((p_acts - acts.to(self.device)) ** 2)

    def clone_policy(self, update_info):
        sampler = self._train_sampler.__iter__()
        for idxes in sampler:
            tic = timeit.default_timer()
            obss, h_states, acts, rews, dones, infos, lengths = self.expert_buffer.sample(self._opt_batch_size, idxes=idxes)
            update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)
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

    def check_overfit(self, update_info):
        with torch.no_grad():
            self.model.eval()
            sampler = self._val_sampler.__iter__()
            total_validation_loss = 0.
            for idxes in sampler:
                tic = timeit.default_timer()
                obss, h_states, acts, rews, dones, infos, lengths = self.expert_buffer.sample(self._opt_batch_size, idxes=idxes)
                update_info[c.VALIDATION_SAMPLE_TIME].append(timeit.default_timer() - tic)
                for accum_i in range(self._accum_num_grad):
                    opt_idxes = range(accum_i * self._num_samples_per_accum,
                                    (accum_i + 1) * self._num_samples_per_accum)
                    validation_loss = self._compute_bc_loss(obss[opt_idxes],
                                                            h_states[opt_idxes],
                                                            acts[opt_idxes],
                                                            lengths[opt_idxes])
                    validation_loss /= self._opt_batch_size
                    total_validation_loss += validation_loss.detach().cpu()
            if total_validation_loss > self.best_validation_loss:
                self._overfit_count += 1
                if self._overfit_count == self._overfit_tolerance:
                    self.overfitted = True
                    self.model.load_state_dict(self._curr_best_model)
            else:
                self._overfit_count = 0
                self.best_validation_loss = total_validation_loss
                update_info[c.VALIDATION_LOSS] = [self.best_validation_loss]
                self._curr_best_model = copy.deepcopy(self.model.state_dict())
            self.model.train()

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.step += 1
        update_info = {}

        update_info[c.BC_LOSS] = []
        update_info[c.SAMPLE_TIME] = []
        update_info[c.POLICY_UPDATE_TIME] = []
        update_info[c.VALIDATION_LOSS] = [self.best_validation_loss]
        update_info[c.VALIDATION_SAMPLE_TIME] = []
        if self.overfitted:
            return False, update_info

        for e in range(self._opt_epochs):
            self.check_overfit(update_info)
            if self.overfitted:
                break
            self.clone_policy(update_info)

            if e % 5 == 0:
                print("Epoch %d | T loss: %.4f | V loss: %.4f | Epochs w/o best: %d | T time: %.5f" %
                      (e, update_info[c.BC_LOSS][-1], update_info[c.VALIDATION_LOSS][-1] / len(self._val_sampler),
                       self._overfit_count, update_info[c.POLICY_UPDATE_TIME][-1]))

        return True, update_info


class MultitaskBC:
    def __init__(self, model, optimizer, expert_buffers, algo_params, aux_tasks=AuxiliaryTask()):
        self._optimizer = optimizer
        self.model = model
        self.expert_buffers = expert_buffers
        self.num_tasks = len(self.expert_buffers)
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
        self._overfit_tolerance = algo_params.get(
            c.OVERFIT_TOLERANCE, c.DEFAULT_BC_PARAMS[c.OVERFIT_TOLERANCE])

        assert self._opt_batch_size % self._accum_num_grad == 0
        self._num_samples_per_accum = self._opt_batch_size // self._accum_num_grad

        self._max_grad_norm = algo_params.get(
            c.MAX_GRAD_NORM, c.DEFAULT_BC_PARAMS[c.MAX_GRAD_NORM])

        self.coefficients = algo_params.get(c.COEFFICIENTS, np.ones(self.num_tasks))

        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]

        self._train_val_ratio = algo_params.get(
            c.VALIDATION_RATIO, c.DEFAULT_BC_PARAMS[c.VALIDATION_RATIO])
        min_size = min([len(buffer) for buffer in self.expert_buffers])

        self._val_samplers = []
        self._train_samplers = []
        self.best_per_task_loss = dict()
        total_size_train_data = 0
        for task_i, expert_buffer in enumerate(self.expert_buffers):
            idxes = np.random.permutation(np.arange(len(expert_buffer)))
            num_val = int(len(expert_buffer) * self._train_val_ratio)
            num_train = len(expert_buffer) - num_val
            self._val_samplers.append(BatchSampler(sampler=SubsetRandomSampler(idxes[num_train:]),
                                                   batch_size=self._opt_batch_size,
                                                   drop_last=False))

            total_size_train_data += len(idxes[:num_train])
            self._train_samplers.append(BatchSampler(sampler=SubsetRandomSampler(idxes[:num_train]),
                                                     batch_size=self._opt_batch_size,
                                                     drop_last=False))
            self.best_per_task_loss[task_i] = np.inf

        dataset_size_reweight = algo_params.get(c.MULTI_BC_DATASET_SIZE_REWEIGHT, True)
        if dataset_size_reweight:
            self._train_ds_size_task_weight = [int(len(buf) * (1 - self._train_val_ratio)) / total_size_train_data \
                                    for buf in expert_buffers]
        else:
            self._train_ds_size_task_weight = [1] * len(expert_buffers)
        self._train_epoch_num_iters = max([len(samp) for samp in self._train_samplers])

        self.best_validation_loss = np.inf
        self._overfit_count = 0
        self.overfitted = False
        self._curr_best_model = copy.deepcopy(self.model.state_dict())

    def state_dict(self):
        state_dict = {}
        state_dict[c.STATE_DICT] = self.model.state_dict()
        state_dict[c.OPTIMIZER] = self._optimizer.state_dict()
        state_dict[c.AUXILIARY_TASKS] = self._aux_tasks.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict[c.STATE_DICT])
        self._optimizer.load_state_dict(state_dict[c.OPTIMIZER])
        self._aux_tasks.load_state_dict(state_dict[c.AUXILIARY_TASKS])

    def _compute_bc_loss(self, obss, h_states, acts, lengths, task_i):
        # old method, not quite correct
        # dist, _, _ = self.model(obss, h_states, lengths=lengths)
        # return torch.sum((dist.mean[:, task_i] - acts.to(self.device)) ** 2)

        p_acts, _ = self.model.deterministic_act_lprob(obss, h_states)
        return torch.sum((p_acts[:, task_i] - acts.to(self.device)) ** 2)

    def clone_policy(self, update_info):
        samplers = [train_sampler.__iter__() for train_sampler in self._train_samplers]

        # make all idxes lists right away
        all_idxes = []
        for t_i in range(self.num_tasks):
            all_task_idxes = []
            while len(all_task_idxes) < self._train_epoch_num_iters:
                sampler = self._train_samplers[t_i].__iter__()
                all_task_idxes.extend(list(sampler))

            all_task_idxes = all_task_idxes[:self._train_epoch_num_iters]
            all_idxes.append(all_task_idxes)

        for train_i in range(self._train_epoch_num_iters):
            self._optimizer.zero_grad()
            idxes = [all_idxes[task_i][train_i] for task_i in range(self.num_tasks)]
            total_bc_loss = 0.

            for task_i, idx in enumerate(idxes):
                tic = timeit.default_timer()
                obss, h_states, acts, rews, dones, infos, lengths = self.expert_buffers[task_i].sample(self._opt_batch_size, idxes=idx)
                update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)

                per_task_loss = 0.
                tic = timeit.default_timer()
                for accum_i in range(self._accum_num_grad):
                    # handle last output of sampler, which will have less than num_samples_per_accum
                    max_idxes = min(len(obss), self._num_samples_per_accum)
                    opt_idxes = range(accum_i * self._num_samples_per_accum,
                                        min((accum_i + 1) * self._num_samples_per_accum, max_idxes))
                    bc_loss = self._compute_bc_loss(obss[opt_idxes],
                                                    h_states[opt_idxes],
                                                    acts[opt_idxes],
                                                    lengths[opt_idxes],
                                                    task_i)
                    bc_loss = bc_loss / self._opt_batch_size * self.coefficients[task_i]
                    bc_loss *= self._train_ds_size_task_weight[task_i]
                    bc_loss.backward()
                    per_task_loss += bc_loss.detach().cpu()
                    total_bc_loss += bc_loss.detach().cpu()
                update_info[f"{c.BC_LOSS}-task_{task_i}"].append(per_task_loss.numpy())

            nn.utils.clip_grad_norm_(self.model.parameters(),
                                    self._max_grad_norm)
            self._optimizer.step()
            update_info[c.BC_LOSS].append(total_bc_loss.numpy())
            update_info[c.POLICY_UPDATE_TIME].append(timeit.default_timer() - tic)

    def check_overfit(self, update_info):
        with torch.no_grad():
            self.model.eval()
            samplers = [val_sampler.__iter__() for val_sampler in self._val_samplers]
            total_validation_loss = 0.
            per_task_loss = {task_i: 0. for task_i in range(len(samplers))}

            self._optimizer.zero_grad()

            # Sample and compute loss from i'th task -- do entire sampler for each task one by one
            for task_i, sampler in enumerate(samplers):
                for idx in sampler:

                    tic = timeit.default_timer()
                    obss, h_states, acts, rews, dones, infos, lengths = self.expert_buffers[task_i].sample(self._opt_batch_size, idxes=idx)
                    update_info[c.VALIDATION_SAMPLE_TIME].append(timeit.default_timer() - tic)

                    tic = timeit.default_timer()
                    for accum_i in range(self._accum_num_grad):
                        # handle last output of sampler, which will have less than num_samples_per_accum
                        max_idxes = min(len(obss), self._num_samples_per_accum)
                        opt_idxes = range(accum_i * self._num_samples_per_accum,
                                        min((accum_i + 1) * self._num_samples_per_accum, max_idxes))
                        validation_loss = self._compute_bc_loss(obss[opt_idxes],
                                                                h_states[opt_idxes],
                                                                acts[opt_idxes],
                                                                lengths[opt_idxes],
                                                                task_i)
                        validation_loss = validation_loss / self._opt_batch_size * self.coefficients[task_i]
                        per_task_loss[task_i] += validation_loss.detach().cpu()
                        total_validation_loss += validation_loss.detach().cpu()

            if total_validation_loss > self.best_validation_loss:
                self._overfit_count += 1
                if self._overfit_count == self._overfit_tolerance:
                    self.overfitted = True
                    self.model.load_state_dict(self._curr_best_model)
            else:
                self._overfit_count = 0
                self.best_validation_loss = total_validation_loss
                self.best_per_task_loss = per_task_loss
                self._curr_best_model = copy.deepcopy(self.model.state_dict())
                update_info[c.VALIDATION_LOSS] = [self.best_validation_loss]
                for task_i in range(len(samplers)):
                    update_info[f"{c.VALIDATION_LOSS}-task_{task_i}"] = [self.best_per_task_loss[task_i]]
            self.model.train()

    def update(self, curr_obs, curr_h_state, act, rew, done, info, next_obs, next_h_state):
        self.step += 1
        update_info = {}

        update_info[c.BC_LOSS] = []
        for task_i in range(self.num_tasks):
            update_info[f"{c.BC_LOSS}-task_{task_i}"] = []
        update_info[c.SAMPLE_TIME] = []
        update_info[c.POLICY_UPDATE_TIME] = []
        update_info[c.VALIDATION_LOSS] = [self.best_validation_loss]
        for task_i in range(len(self.expert_buffers)):
            update_info[f"{c.VALIDATION_LOSS}-task_{task_i}"] = [self.best_per_task_loss[task_i]]
        update_info[c.VALIDATION_SAMPLE_TIME] = []
        if self.overfitted:
            return False, update_info

        for e in range(self._opt_epochs):
            self.check_overfit(update_info)
            if self.overfitted:
                break
            self.clone_policy(update_info)

            if e % 5 == 0:
                print("Epoch %d | T loss: %.4f | V loss: %.4f | Epochs w/o best: %d | T time: %.5f" %
                      (e, update_info[c.BC_LOSS][-1], update_info[c.VALIDATION_LOSS][-1],
                       self._overfit_count, update_info[c.POLICY_UPDATE_TIME][-1]))

        return True, update_info
