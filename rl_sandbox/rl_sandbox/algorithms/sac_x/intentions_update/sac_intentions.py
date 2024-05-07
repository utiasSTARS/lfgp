import timeit
import torch
import numpy as np
import copy

import rl_sandbox.constants as c

from rl_sandbox.algorithms.dac.sac import SACDAC
from rl_sandbox.algorithms.utils import aug_data
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask

from rl_sandbox.model_architectures.actor_critics.fully_connected_soft_actor_critic import \
    MultiTaskFullyConnectedSquashedGaussianTD3

def timer(): return timeit.default_timer()

class UpdateSACIntentions(SACDAC):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        super().__init__(model=model,
                         policy_opt=policy_opt,
                         qs_opt=qs_opt,
                         alpha_opt=alpha_opt,
                         learn_alpha=learn_alpha,
                         buffer=buffer,
                         algo_params=algo_params,
                         aux_tasks=aux_tasks)

        self._num_tasks = algo_params.get(c.NUM_TASKS, 1)

        # calculate per task loss weights
        self._main_task_loss_weight = algo_params.get(c.MAIN_TASK_LOSS_WEIGHT, 1.0)
        aux_task_weight = self._num_tasks / (self._num_tasks - 1 + self._main_task_loss_weight)
        true_main_task_weight = self._main_task_loss_weight * aux_task_weight
        self._task_loss_weight = torch.ones(self._num_tasks, device=self.device, requires_grad=False)
        self._task_loss_weight[algo_params[c.MAIN_INTENTION]] = true_main_task_weight
        aux_intentions = list(range(self._num_tasks))
        aux_intentions.remove(algo_params[c.MAIN_INTENTION])
        self._task_loss_weight[aux_intentions] = aux_task_weight

    def _compute_qs_loss(self, obss, h_states, acts, rews, dones, next_obss, discounting, lengths, update_info,
                         n_obss=None, n_h_states=None, n_discounts=None, discount_includes_gamma=True):
        batch_size = len(obss)

        # get q vals for all tasks
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        max_expert_q = torch.zeros(self._num_tasks, device=self.device)  # to be used for q over max penalty in sqil
        avg_expert_q = torch.zeros(self._num_tasks, device=self.device)  # to be used for q over max penalty in sqil
        for task_i in range(self._num_tasks):
            if self._total_expert_amount > 0:
                start = self._expert_starts[task_i]
                end = self._expert_ends[task_i]
                max_exp_q = q1_val[start:end, task_i].max().detach()
                avg_exp_q = q1_val[start:end, task_i].mean().detach()
                update_info[f"{c.AVG_EXPERT_Q}/task_{task_i}"].append(avg_exp_q.cpu().numpy())
                update_info[f"{c.MAX_EXPERT_Q}/task_{task_i}"].append(max_exp_q.cpu().numpy())
                max_expert_q[task_i] = max_exp_q
                avg_expert_q[task_i] = avg_exp_q

            update_info[f"{c.MAX_POLICY_Q}/task_{task_i}"].append(q1_val[self._total_expert_amount:, task_i].max().detach().cpu().numpy())
            update_info[f"{c.AVG_POLICY_Q}/task_{task_i}"].append(q1_val[self._total_expert_amount:, task_i].mean().detach().cpu().numpy())

        # set tasks, dones, and discounting to have shapes matching what outputs from q functions will be

        # if self._expert_buffer_sampling and not self.self._expert_buf_critic_share_all:
        #     raise NotImplementedError("TODO")
        #     # TODO all of the below will be necessary if we want to use this for trajectory based expert data,
        #     # but for final tasks only, expert data does not need to bootstrap
        #     # first buf_samp_batch_size are for task 0, next for task 1, ... last for all tasks, b/c it is non-expert data
        #     non_expert_start_idx = sum(self._expert_amounts)  # also is the number of expert and non-expert data
        #     task_batch_size = (batch_size - self._buffer_sample_batch_size) + self._buffer_sample_batch_size * self._num_tasks
        #     if task_batch_size % self._num_tasks != 0:
        #         print("something has gone wrong with the batch size..")
        #         import ipdb; ipdb.set_trace()
        #     tasks = torch.arange(self._num_tasks, device=self.device).repeat(int(task_batch_size / self._num_tasks))

        #     # first set expert portions using only corresponding data
        #     corresponding_rews = torch.zeros(task_batch_size, 1)
        #     corresponding_dones = torch.zeros(task_batch_size, 1)
        #     corresponding_discounting = torch.zeros(task_batch_size, 1)
        #     for task_i, (start, end) in enumerate(zip(self._expert_starts, self._expert_ends)):
        #         corresponding_rews[start:end] = rews[start:end, task_i][:, None] * self._reward_scaling
        #         if dones.shape[1] != self._num_tasks:
        #             corresponding_dones[start:end] = dones[start:end]
        #         else:
        #             corresponding_dones[start:end] = dones[start:end, task_i][:, None]
        #         corresponding_discounting[start:end] = discounting[start:end]

        #     # now set all non-expert parts with everything
        #     corresponding_rews[non_expert_start_idx:] = \
        #         rews[non_expert_start_idx:].reshape(non_expert_start_idx, 1) * self._reward_scaling
        #     if dones.shape[1] != self._num_tasks:
        #         corresponding_dones[non_expert_start_idx:] = \
        #             dones[non_expert_start_idx:].repeat(1, self._num_tasks).reshape(non_expert_start_idx, 1)
        #     else:
        #         corresponding_dones[non_expert_start_idx:] = dones[non_expert_start_idx:].reshape(non_expert_start_idx, 1)
        #     corresponding_discounting[non_expert_start_idx:] = \
        #         discounting[non_expert_start_idx:].repeat(1, self._num_tasks).reshape(non_expert_start_idx, 1)

        #     rews, dones, discounting = corresponding_rews, corresponding_dones, corresponding_discounting
        #     alpha = self.model.alpha.detach().repeat(int(task_batch_size / self._num_tasks))

        task_batch_size = batch_size * self._num_tasks

        tasks = torch.arange(self._num_tasks, device=self.device).repeat(batch_size).reshape(task_batch_size, 1)
        rews = rews.reshape(task_batch_size, 1) * self._reward_scaling

        # dones will generally be not per task, but we can manually set it to be per task
        # this is useful for how SQIL is formulated
        if dones.shape[1] != self._num_tasks:
            dones = dones.repeat(1, self._num_tasks).reshape(task_batch_size, 1)
        else:
            dones = dones.reshape(task_batch_size, 1)

        discounting = discounting.repeat(1, self._num_tasks).reshape(task_batch_size, 1)

        alpha = self.model.alpha.detach().repeat(batch_size).reshape(task_batch_size, 1)

        rews, dones, discounting = rews.to(self.device), dones.to(self.device), discounting.to(self.device)

        with torch.no_grad():

            # this is an unnecessary forward pass, so removing
            # _, _, _, targ_next_h_states = self._target_model.q_vals(obss, h_states, acts, lengths=lengths)
            targ_next_h_states = h_states

            if n_obss is None:
                if isinstance(self.model, MultiTaskFullyConnectedSquashedGaussianTD3):
                    next_acts, next_lprobs = self.model.act_lprob(
                        next_obss, next_h_states, clip=self.model._no_squash_act_clip)
                else:
                    next_acts, next_lprobs = self.model.act_lprob(next_obss, next_h_states)

                # if self._expert_buffer_sampling and self._expert_buf_critic_share_all:
                #     # only take the actions from policies corresponding to relevant data
                #     # current shape is batch_size x num_tasks x act_dim
                #     # batch_size * num_tasks for expert data, batch_size * num_tasks for policy data
                #     corresponding_next_acts = torch.zeros(task_batch_size, acts.shape[-1], device=self.device)
                #     corresponding_next_lprobs = torch.zeros(task_batch_size, 1, device=self.device)
                #     # corresponding_next_obss =

                #     # first get the expert ones
                #     for task_i, (start, end) in enumerate(zip(self._expert_starts, self._expert_ends)):
                #         corresponding_next_acts[start:end] = next_acts[start:end, task_i]
                #         corresponding_next_lprobs[start:end] = next_lprobs[start:end, task_i]

                #     # now the non-expert ones
                #     corresponding_next_acts[non_expert_start_idx:] = \
                #         next_acts[non_expert_start_idx:].reshape(non_expert_start_idx, self._action_dim)
                #     corresponding_next_lprobs[non_expert_start_idx:] = \
                #         next_lprobs[non_expert_start_idx:].reshape(non_expert_start_idx, 1)
                #     next_acts = corresponding_next_acts
                #     next_lprobs = corresponding_next_lprobs

                #     # also need to resize next_obss and targ_next_h_states
                #     # TODO let's do a better re-factor here, since for this work we know that we don't need
                #     # to bootstrap off of expert data, since we don't have real acts or next_obss for them
                #     import ipdb; ipdb.set_trace()

                #     # TODO don't forget to divide q losses by 2, since they'll be twice as big as the new batch size

                next_acts = next_acts.reshape(task_batch_size, self._action_dim)
                next_lprobs = next_lprobs.reshape(task_batch_size, 1)

                min_q_targ, _, _, _ = self._target_model.q_vals(
                    aug_data(data=next_obss, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
                    aug_data(data=targ_next_h_states, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
                    next_acts)

                min_q_targ = torch.gather(min_q_targ, dim=1, index=tasks)
                min_q_targ = min_q_targ.detach()

            else:

                all_next_obss = torch.cat([next_obss, n_obss])
                all_next_h_states = torch.cat([next_h_states, n_h_states])

                if isinstance(self.model, MultiTaskFullyConnectedSquashedGaussianTD3):
                    all_next_acts, all_next_lprobs = self.model.act_lprob(
                        all_next_obss, all_next_h_states, clip=self.model._no_squash_act_clip)
                else:
                    all_next_acts, all_next_lprobs = self.model.act_lprob(all_next_obss, all_next_h_states)

                all_next_acts = all_next_acts.reshape(task_batch_size * 2, self._action_dim)
                all_next_lprobs = all_next_lprobs.reshape(task_batch_size * 2, 1)

                all_targnexth = torch.cat([targ_next_h_states, n_h_states])

                _, all_q1_targ, all_q2_targ, _ = self._target_model.q_vals(
                    aug_data(data=all_next_obss, num_aug=self._num_tasks, aug_batch_size=task_batch_size*2),
                    aug_data(data=all_targnexth, num_aug=self._num_tasks, aug_batch_size=task_batch_size*2),
                    all_next_acts)

                next_lprobs = all_next_lprobs[:task_batch_size]
                n_lprobs = all_next_lprobs[task_batch_size:]
                q1_targ_next = all_q1_targ[:task_batch_size]
                q1_targ_n = all_q1_targ[task_batch_size:]
                q2_targ_next = all_q2_targ[:task_batch_size]
                q2_targ_n = all_q2_targ[task_batch_size:]

                assert len(n_discounts.shape) == 2, "command below won't work if there's no dummy dimension on the end"
                n_discounts = n_discounts.repeat(1, self._num_tasks).reshape(task_batch_size, 1)
                q1_targ = self.algo_params.get(c.NTH_Q_TARG_MULTIPLIER, .5) * (q1_targ_next + n_discounts * q1_targ_n)
                q2_targ = self.algo_params.get(c.NTH_Q_TARG_MULTIPLIER, .5) * (q2_targ_next + n_discounts * q2_targ_n)

                min_q_targ = torch.min(q1_targ, q2_targ)
                min_q_targ = torch.gather(min_q_targ, dim=1, index=tasks)
                min_q_targ = min_q_targ.detach()

                l_probs_combined = self.algo_params.get(c.NTH_Q_TARG_MULTIPLIER, .5) * \
                        (next_lprobs + n_discounts * n_lprobs)
                next_lprobs = l_probs_combined

            if isinstance(self.model, MultiTaskFullyConnectedSquashedGaussianTD3) or self._no_entropy_in_qloss:
                # TD3 without major refactor
                v_next = min_q_targ
            else:
                v_next = (min_q_targ - alpha * next_lprobs)

            if discount_includes_gamma:
                discount = discounting
            else:
                discount = self._gamma * discounting

            if self._reward_model == 'rce':
                assert self.model._classifier_output, "Need a classifier output on network for RCE to work!"

                # add numerical stability
                v_next = torch.clip(v_next, min=self._rce_eps, max=1-self._rce_eps)

                w = v_next / (1 - v_next)
                td_targets = discount * w / (discount * w + 1)
                weights = 1 + discount * w
                td_targets = td_targets * (1 - dones)

                # now set expert indices to ones, all others are just the targets above
                # to have same behaviour as original RCE, sqil_rce_bootstrap_dones must be off (which sets all expert dones to 1)
                td_targets_by_task = td_targets.reshape(-1, self._num_tasks)
                weights_by_task = weights.reshape(-1, self._num_tasks)
                dones_by_task = dones.reshape(-1, self._num_tasks)
                discount_by_task = discount.reshape(-1, self._num_tasks)
                for task_i, (start, end) in enumerate(zip(self._expert_starts, self._expert_ends)):
                    td_targets_by_task[start:end, task_i] = \
                        1.0 + (1 - dones_by_task[start:end, task_i]) * td_targets_by_task[start:end, task_i]
                    weights_by_task[start:end, task_i] = 1 - discount_by_task[start:end, task_i]

                target = td_targets_by_task.reshape(task_batch_size, 1)
                weights = weights_by_task.reshape(task_batch_size, 1)
            else:
                target = rews + discount * (1 - dones) * v_next
                weights = torch.ones_like(target)

            if hasattr(self.model, c.VALUE_RMS):
                target = target.cpu().reshape(batch_size, self._num_tasks)
                self.model.value_rms.update(target)
                target = self.model.value_rms.normalize(target)
                target = target.to(self.device).reshape(task_batch_size, 1)

        # reshape weights to be by task
        weights_by_task = weights.reshape(-1, self._num_tasks)

        if self._expert_buffer_sampling and not self._expert_buf_critic_share_all:
            weights_by_task[:self._total_expert_amount][self._invalid_expert] = 0.0

        # multiply by hyperparameter per task loss weight
        weights_by_task *= self._task_loss_weight

        # back to flattened
        weights = weights_by_task.reshape(task_batch_size, 1)

        if self.model._classifier_output:
            assert self.algo_params.get(c.SQIL_RCE_BOOTSTRAP_EXPERT_MODE, "no_boot") == "no_boot", \
                "Cannot bootstrap on expert done if using classifier output."
            classify_loss = torch.nn.BCELoss(weight=weights, reduction="sum")
            q1_loss = classify_loss(q1_val.reshape(task_batch_size, 1), target)
            q2_loss = classify_loss(q2_val.reshape(task_batch_size, 1), target)
        else:
            q1_loss_all = weights * (q1_val.reshape(task_batch_size, 1) - target) ** 2
            q2_loss_all = weights * (q2_val.reshape(task_batch_size, 1) - target) ** 2
            q1_loss = q1_loss_all.sum()
            q2_loss = q2_loss_all.sum()

        if self.algo_params.get(c.Q_OVER_MAX_PENALTY, 0.0) > 0:
            penalty = self.algo_params[c.Q_OVER_MAX_PENALTY]
            discount_by_task = discount.reshape(-1, self._num_tasks)
            num_med_filt = self.algo_params.get(c.QOMP_NUM_MED_FILT, 50)

            if self._reward_model == c.SQIL or self._reward_model == c.SPARSE:
                # q over max penalty based on things either having reward 1 or sqil reward label (0 or -1)
                q_max = self._reward_scaling * torch.ones_like(q1_val) / (1 - discount_by_task)
                q_min = self._reward_scaling * torch.ones_like(q1_val) / (1 - discount_by_task) * \
                    self.algo_params.get(c.SQIL_POLICY_REWARD_LABEL, 0.0)

            elif self._reward_model == c.DISCRIMINATOR:
                # compute a q mag penalty based on minimum and maximum reward in the batch
                rews_by_task = rews.reshape(-1, self._num_tasks)
                min_r = torch.minimum(torch.tensor(-.01), rews_by_task.min(axis=0)[0])
                max_r = torch.maximum(torch.tensor(.01), rews_by_task.max(axis=0)[0])

                if not hasattr(self, "_prev_maxs"):
                    self._prev_maxs = torch.ones((num_med_filt, self._num_tasks), device=self.device) * max_r
                if not hasattr(self, "_prev_mins"):
                    self._prev_mins = torch.ones((num_med_filt, self._num_tasks), device=self.device) * min_r

                self._prev_mins = self._prev_mins.roll(1, dims=0)
                self._prev_mins[0] = min_r
                self._prev_maxs = self._prev_maxs.roll(1, dims=0)
                self._prev_maxs[0] = max_r

                min_r_filtered = self._prev_mins.median(axis=0)[0]
                max_r_filtered = self._prev_maxs.median(axis=0)[0]
                q_min = min_r_filtered * torch.ones_like(q1_val) / (1 - discount_by_task)
                q_max = max_r_filtered * torch.ones_like(q1_val) / (1 - discount_by_task)

            else:
                raise NotImplementedError(f"Q over max penalty not implemented for reward model {self._reward_model}")

            # also set q max for policy to be even more restrictive, based on whatever the current q max for expert is
            if self._total_expert_amount > 0:
                max_q = max_expert_q if self.algo_params.get(c.QOMP_POLICY_MAX_TYPE, 'max_exp') == 'max_exp' else avg_expert_q
                if not hasattr(self, "_prev_q_maxs"):
                    self._prev_q_maxs = torch.ones((num_med_filt, self._num_tasks), device=self.device) * max_q
                self._prev_q_maxs = self._prev_q_maxs.roll(1, dims=0)
                self._prev_q_maxs[0] = max_q
                max_exp_q_filtered = self._prev_q_maxs.median(axis=0)[0]
                q_max[self._total_expert_amount:, :] = max_exp_q_filtered

            q1_max_mag_loss = penalty * torch.maximum(q1_val - q_max, torch.tensor(0)) ** 2
            q2_max_mag_loss = penalty * torch.maximum(q2_val - q_max, torch.tensor(0)) ** 2
            q1_min_mag_loss = penalty * torch.maximum(-(q1_val - q_min), torch.tensor(0)) ** 2
            q2_min_mag_loss = penalty * torch.maximum(-(q2_val - q_min), torch.tensor(0)) ** 2

            q1_loss = q1_loss + q1_max_mag_loss.sum() + q1_min_mag_loss.sum()
            q2_loss = q2_loss + q2_max_mag_loss.sum() + q2_min_mag_loss.sum()

        return q1_loss, q2_loss

    def _compute_pi_loss(self, obss, h_states, acts, lengths):
        batch_size = len(obss)
        task_batch_size = batch_size * self._num_tasks

        tasks = torch.arange(self._num_tasks, device=self.device).repeat(batch_size).reshape(task_batch_size, 1)

        if isinstance(self.model, MultiTaskFullyConnectedSquashedGaussianTD3):
            acts, lprobs, raw_acts = self.model.act_lprob(obss, h_states, clip=self.model._no_squash_act_clip,
                                                          return_raw_act=True)
        else:
            acts, lprobs, raw_acts = self.model.act_lprob(obss, h_states, return_raw_act=True)

        acts = acts.reshape(task_batch_size, self._action_dim)
        lprobs = lprobs.reshape(task_batch_size, 1)

        min_q, _, _, _ = self.model.q_vals(
            aug_data(data=obss, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
            aug_data(data=h_states, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
            acts,
            lengths=lengths.repeat(1, self._num_tasks).reshape(task_batch_size))
        min_q = torch.gather(min_q, dim=1, index=tasks)

        alpha = self.model.alpha.detach().repeat(batch_size).reshape(task_batch_size, 1)

        if self._expert_buffer_sampling and not self._expert_buf_policy_share_all and not self._expert_buffer_model_no_policy:
            # don't update policy based on non-corresponding expert data
            min_q *= self._valid_all
            lprobs *= self._valid_all

        task_loss_weights = self._task_loss_weight.repeat(batch_size).reshape(task_batch_size, 1)

        if isinstance(self.model, MultiTaskFullyConnectedSquashedGaussianTD3):
            # TD3 without major refactor
            pi_loss = (task_loss_weights * (-min_q)).sum()
        else:
            pi_loss = (task_loss_weights * (alpha * lprobs - min_q)).sum()

        # add a raw action magnitude loss to discourage overfitting individual dimensions
        pi_loss += self.algo_params[c.ACTOR_RAW_MAGNITUDE_PENALTY] * torch.square(raw_acts).sum()

        return pi_loss

    def _compute_alpha_loss(self, obss, h_states, lengths):
        batch_size = len(obss)
        task_batch_size = batch_size * self._num_tasks
        with torch.no_grad():
            _, lprobs = self.model.act_lprob(obss, h_states, lengths=lengths)
            lprobs = lprobs.reshape(task_batch_size, 1)

        alpha = self.model.alpha.repeat(batch_size, 1).reshape(task_batch_size, 1)

        if self._expert_buffer_sampling and not self._expert_buf_policy_share_all and not self._expert_buffer_model_no_policy:
            # don't update alpha based on non-corresponding expert data
            alpha *= self._valid_all

        task_loss_weights = self._task_loss_weight.repeat(batch_size).reshape(task_batch_size, 1)

        alpha_loss = (task_loss_weights * (-alpha * (lprobs + self._target_entropy).detach())).sum()

        return alpha_loss


class UpdateSACDACIntentions(UpdateSACIntentions):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask(),
                 expert_buffers=None):
        super().__init__(model=model,
                         policy_opt=policy_opt,
                         qs_opt=qs_opt,
                         alpha_opt=alpha_opt,
                         learn_alpha=learn_alpha,
                         buffer=buffer,
                         algo_params=algo_params,
                         aux_tasks=aux_tasks)

        self.expert_buffers = expert_buffers
        self._expert_buffer_rate = self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_RATE, 0.)
        self._expert_buf_critic_share_all = self.algo_params.get(c.EXPERT_BUFFER_CRITIC_SHARE_ALL, True)
        self._expert_buf_policy_share_all = self.algo_params.get(c.EXPERT_BUFFER_POLICY_SHARE_ALL, True)
        self._expert_starts = None
        self._expert_ends = None
        self._expert_amounts = None
        self._total_expert_amount = 0
        self._expert_buffer_sampling = False
        self._expert_buffer_size_type = self.algo_params.get(c.EXPERT_BUFFER_SIZE_TYPE, "fraction")

        # if not self._expert_buf_critic_share_all and self._expert_buffer_rate > 0.0:
        if self._expert_buffer_size_type == "match_bs" and self._expert_buffer_rate > 0.0:
            # now batch sizes are much bigger since we append equal amounts of expert data per task to buffer samples
            self._batch_size = self._batch_size * (self._num_tasks + 1)
            self._num_samples_per_accum = self._batch_size // self._accum_num_grad

    def update(self, reward_function, next_obs, next_h_state):
        self.step += 1

        update_info = {}

        # Perform SAC update
        if self.step >= self._buffer_warmup and self.step % self._steps_between_update == 0:
            update_info[c.PI_LOSS] = []
            update_info[c.Q1_LOSS] = []
            update_info[c.Q2_LOSS] = []
            update_info[c.ALPHA] = []
            update_info[c.SAMPLE_TIME] = []
            update_info[c.Q_UPDATE_TIME] = []
            update_info[c.POLICY_UPDATE_TIME] = []
            update_info[c.ALPHA_LOSS] = []
            update_info[c.ALPHA_UPDATE_TIME] = []
            for task_i in range(self._num_tasks):
                update_info[f"{c.DISCRIMINATOR_REWARD}/task_{task_i}"] = []
                update_info[f"{c.MAX_POLICY_Q}/task_{task_i}"] = []
                update_info[f"{c.AVG_EXPERT_Q}/task_{task_i}"] = []
                update_info[f"{c.MAX_EXPERT_Q}/task_{task_i}"] = []
                update_info[f"{c.AVG_POLICY_Q}/task_{task_i}"] = []

            self._expert_buffer_sampling = False

            for _ in range(self._num_gradient_updates // self._num_prefetch):
                tic = timeit.default_timer()

                # sample observations from n step buffer or regular buffer
                if self._sample_horizon > 2:
                    obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, sample_idxes, ep_lengths =\
                        self.buffer.sample_trajs(self._buffer_sample_batch_size * self._num_prefetch, None, None)

                else:
                    obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, sample_idxes = \
                        self.buffer.sample_with_next_obs(
                            self._buffer_sample_batch_size * self._num_prefetch, next_obs, next_h_state,
                            include_random_idxes=True)

                # replace obss/acts with expert data if expert_buffer_rate is set
                decay = self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_DECAY, 1.0)
                self._expert_buffer_rate = self._expert_buffer_rate * decay

                if int(self._expert_buffer_rate * self._batch_size * self._num_prefetch) > 0 or \
                    self._reward_model in ['sqil', 'rce']:
                    ##### START expert buffer sampling

                    assert self.expert_buffers is not None
                    self._expert_buffer_sampling = True

                    num_expert_total = int(self._expert_buffer_rate * self._batch_size * self._num_prefetch)
                    num_policy = int(self._batch_size * self._num_prefetch - num_expert_total)
                    num_per_buffer = int(num_expert_total / len(self.expert_buffers))
                    num_main = num_expert_total - (len(self.expert_buffers) - 1) * num_per_buffer
                    amounts = [num_per_buffer] * len(self.expert_buffers)
                    amounts[self.algo_params[c.MAIN_INTENTION]] = num_main

                    if self._expert_buffer_size_type == "match_bs":
                        obss = obss.repeat(self._num_tasks + 1, 1, 1)
                        next_obss = next_obss.repeat(self._num_tasks + 1, 1, 1)
                        h_states = h_states.repeat(self._num_tasks + 1, 1, 1)
                        next_h_states = next_h_states.repeat(self._num_tasks + 1, 1, 1)
                        acts = acts.repeat(self._num_tasks + 1, 1)
                        dones = dones.repeat(self._num_tasks + 1, 1)
                        for k in infos:
                            infos[k] = infos[k].repeat(self._num_tasks + 1, 1)
                        lengths = lengths.repeat(self._num_tasks + 1)
                        rews = rews.repeat(self._num_tasks + 1, 1)
                        sample_idxes = sample_idxes.repeat(self._num_tasks + 1)

                        num_expert_total = int(self._num_tasks * self._buffer_sample_batch_size * self._num_prefetch)
                        amounts = [self._buffer_sample_batch_size] * self._num_tasks

                    inds_to_update = np.cumsum(amounts)
                    starts = np.concatenate([[0], inds_to_update[:-1]])
                    ends = inds_to_update

                    self._expert_starts = starts
                    self._expert_ends = ends
                    self._expert_amounts = amounts
                    self._total_expert_amount = sum(amounts)

                    # generate a mask for which data to ignore
                    if not self._expert_buf_critic_share_all or not self._expert_buf_policy_share_all:
                        self._invalid_expert = torch.ones([self._total_expert_amount, self._num_tasks], dtype=torch.bool,
                                                          requires_grad=False)

                    for task_i, (start, end, amount) in enumerate(zip(starts, ends, amounts)):
                        if self._sample_horizon > 2:
                            e_obss, e_h_states, e_acts, e_rews, e_dones, e_next_obss, e_next_h_states, e_infos, \
                            e_lengths, e_sample_idxes, e_ep_lengths = \
                                self.expert_buffers[task_i].sample_trajs(amount, None, None)

                            ep_lengths[start:end] = e_ep_lengths
                            # removing entirely from here because it's overwritten anyways, and it causes problems if
                            # the expert data was collected using an expert with a different reward dimension.
                            # rews[start:end] = e_rews[:, :, :self.algo_params['num_tasks']]
                        else:
                            e_obss, e_h_states, e_acts, e_rews, e_dones, e_next_obss, e_next_h_states, e_infos, \
                                e_lengths, e_sample_idxes = \
                                self.expert_buffers[task_i].sample_with_next_obs(amount, None, None, include_random_idxes=True)

                            # these might not be the correct reward indices, but they get overwritten anyways so doesn't matter for now
                            # specifically, if the expert buffer reward dimension is diff from the env, which can happen
                            # if we're reusing data but removing one of the intentions, these rewards will be wrong
                            # instead, just going to remove this entirely
                            # rews[start:end] = e_rews[:, :self.algo_params['num_tasks']]

                        obss[start:end] = e_obss
                        h_states[start:end] = e_h_states
                        acts[start:end] = e_acts
                        dones[start:end] = e_dones
                        next_obss[start:end] = e_next_obss
                        next_h_states[start:end] = e_next_h_states
                        for k in infos:
                            if k in e_infos.keys():
                                infos[k][start:end] = e_infos[k]
                        lengths[start:end] = e_lengths
                        sample_idxes[start:end] = e_sample_idxes

                        # duplicate obs if we don't have full trajectories
                        if self.algo_params.get(c.EXPERT_DATA_MODE, 'obs_act') == 'obs_only_no_next':
                            # next obs is overwritten with current obs
                            if self._obs_only_no_next_add_noise:
                                # in this case the observation already has noise, so we'll add noise to the original obs
                                # again by resampling
                                e_obss_resampled, _, _, _, _, _, _, _, _ = \
                                    self.expert_buffers[task_i].sample_with_next_obs(
                                        amount, None, None, idxes=e_sample_idxes)

                                next_obss[start:end] = e_obss_resampled
                            else:

                                next_obss[start:end] = obss[start:end]

                            next_h_states[start:end] = h_states[start:end]

                        # set invalid indices if necessary
                        if not self._expert_buf_critic_share_all or not self._expert_buf_policy_share_all:
                            self._invalid_expert[start:end, task_i] = 0

                    if not self._expert_buf_critic_share_all or not self._expert_buf_policy_share_all:
                        self._valid_all = torch.ones([self._batch_size, self._num_tasks], dtype=torch.bool,
                                                     requires_grad=False, device=self.device)
                        self._valid_all[:self._total_expert_amount] = ~self._invalid_expert
                        self._valid_all = self._valid_all.reshape(-1, 1)

                    # replace actions if we're doing obs only expert data
                    all_exp_inds_sl = slice(starts[0], ends[-1])
                    if 'obs_only' in self.algo_params.get(c.EXPERT_DATA_MODE, 'obs_act'):
                        # overwrite actions with mean policy output instead...RCE uses samples from policy
                        # we'll try samples first, and then later with mean
                        # see https://github.com/google-research/google-research/blob/ccf75222d6c63dc16a4e195d2b2223d8e1a160b4/rce/rce_agent.py#L547
                        if self._sample_horizon > 2:
                            exp_obs, exp_h_states = obss[all_exp_inds_sl], h_states[all_exp_inds_sl]

                            # pass all n steps through net simultaneously
                            obs_flat_along_n = exp_obs.reshape(-1, exp_obs.shape[-1])
                            h_states_flat_along_n = exp_obs.reshape(-1, exp_h_states.shape[-1])
                            with torch.no_grad():
                                cur_pol_exp_obs_acts_flat, _ = self.model.act_lprob(obs_flat_along_n,
                                                                                    h_states_flat_along_n)

                            cur_pol_exp_obs_acts = cur_pol_exp_obs_acts_flat.reshape(
                                exp_obs.shape[0], exp_obs.shape[1], cur_pol_exp_obs_acts_flat.shape[1],
                                cur_pol_exp_obs_acts_flat.shape[2])

                            for task_i, (start, end, amount) in enumerate(zip(starts, ends, amounts)):
                                acts[start:end] = cur_pol_exp_obs_acts[start:end, :, task_i]

                        else:
                            with torch.no_grad():
                                cur_pol_exp_obs_acts, _ = self.model.act_lprob(obss[all_exp_inds_sl],
                                                                                h_states[all_exp_inds_sl])

                            # cur_pol_exp_obs_acts now has outputs for all policies, we only want matched ones
                            # separate for loop from above to do just a single net forward pass
                            for task_i, (start, end, amount) in enumerate(zip(starts, ends, amounts)):
                                acts[start:end] = cur_pol_exp_obs_acts[start:end, task_i]

                    ##### END expert buffer sampling

                obss = self.train_preprocessing(obss)
                next_obss = self.train_preprocessing(next_obss)

                # set correct variables if we're using n step
                if self._sample_horizon > 2:

                    # precalculate discount products for all n steps
                    infos[c.DISCOUNTING] = infos[c.DISCOUNTING].squeeze()
                    dis_prod = torch.cumprod(infos[c.DISCOUNTING], dim=1)
                    full_dis_prod = dis_prod * self._gamma_horizon  # element wise

                    all_next_obss = next_obss.clone()
                    all_next_h_states = next_h_states.clone()
                    if self._n_step_mode == 'n_rew_only' or self._n_step_mode == 'nth_q_targ':
                        # we could use the ones from obss, but this misses final observations + next observations
                        # next_obss = obss[:, 1].unsqueeze(1)
                        # next_h_states = h_states[:, 1].unsqueeze(1)
                        next_obss = next_obss[:, 0].unsqueeze(1)
                        next_h_states = next_h_states[:, 0].unsqueeze(1)
                    else:
                        next_obss = obss[:, -1].unsqueeze(1)
                        next_h_states = h_states[:, -1].unsqueeze(1)

                    # keep names consistent for training with nstep on/off
                    all_obss = obss.clone()
                    all_h_states = h_states.clone()
                    all_acts = acts.clone()
                    obss = obss[:, 0].unsqueeze(1)
                    h_states = h_states[:, 0].unsqueeze(1)
                    acts = acts[:, 0]
                    dones = dones[:, 0, :]

                if hasattr(self.model, c.OBS_RMS):
                    self.model.obs_rms.update(obss)

                # if we want to bootstrap even if "done", set all dones to zero
                # done here so that we can still overwrite this for e.g. SQIL rewards
                if self._bootstrap_on_done:
                    dones = torch.zeros_like(dones)

                # compute the new reward using the provided reward function (discriminator or otherwise)
                with torch.no_grad():
                    if self._reward_model == c.DISCRIMINATOR:
                        # nth_q_targ is the version from RCE that doesn't use reward for n step
                        if self._sample_horizon > 2 and not self._n_step_mode == 'nth_q_targ':
                            # sample horizon over 2 indicates n step
                            all_obss_flat = all_obss.clone().reshape(-1, all_obss.shape[-1])
                            all_acts_flat = all_acts.clone().reshape(-1, all_acts.shape[-1])

                            all_rews_flat = reward_function(all_obss_flat, all_acts_flat)
                            all_rews = all_rews_flat.reshape(all_obss.shape[0], self._sample_horizon, -1)

                            if self._n_step_mode == 'n_rew_only':
                                # only taking current reward + n-th step
                                discounts = full_dis_prod[torch.arange(full_dis_prod.shape[0]), ep_lengths]
                                rews = all_rews[:, 0].clone()
                                rews += (discounts * all_rews[torch.arange(all_rews.shape[0]), ep_lengths].T).T
                            elif self._n_step_mode == 'sum_pad':
                                # calcuate reward by summing + multiplying
                                rews = torch.einsum('ijk,ij->ik', all_rews, full_dis_prod).detach()
                            else:
                                raise NotImplementedError(f"Not immplemented for n step mode {self._n_step_mode}")

                        else:
                            # this idxes line is an RNN thing that doesn't actually do anything if not using RNNs
                            idxes = lengths.unsqueeze(-1).repeat(1, *obss.shape[2:]).unsqueeze(1)
                            last_obss = torch.gather(obss, axis=1, index=idxes - 1)[:, 0, :]
                            rews = reward_function(last_obss, acts).detach()

                    elif self._reward_model == c.SQIL or self._reward_model == c.RCE:
                        # SQIL: rewards are 1 for exp data and 0 for all others, done per intention
                        # the way SQIL is implemented in the RCE paper, the actual expert targets are directly labelled
                        # with 1.0, with no bootstrapping, i.e. as if the expert data is trained with done

                        # for RCE, we'll just use the same logic for dones since we end up ignoring the rewards anyways
                        assert self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_DECAY, 1.0) == 1.0, \
                            "expert buffer sampling must have no decay for sqil"
                        assert self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_RATE, 0.) == 0.5, \
                            "Sample rate of 0.5 matches SQIL"

                        if self._sample_horizon > 2:
                            assert self._n_step_mode == 'nth_q_targ', "use nth_q_targ to match RCE implementation if using nstep"
                            rews = rews[:, 0]  # since we still have n-step sampled rewards

                        # make dones per task so we can handle the bootstrapping issue as described above
                        sqil_bootstrap_expert_dones = self.algo_params.get(c.SQIL_RCE_BOOTSTRAP_EXPERT_MODE, "no_boot") == "boot"
                        if not sqil_bootstrap_expert_dones:
                            dones = dones.repeat(1, self._num_tasks)
                        for task_i, (start, end, amount) in enumerate(zip(starts, ends, amounts)):
                            rews[start:end, task_i] = 1.0

                            if self.model._classifier_output:
                                rews[:start, task_i] = 0.0
                                rews[end:, task_i] = 0.0
                            else:
                                rews[:start, task_i] = self.algo_params.get(c.SQIL_POLICY_REWARD_LABEL, 0.0)
                                rews[end:, task_i] = self.algo_params.get(c.SQIL_POLICY_REWARD_LABEL, 0.0)
                            if not sqil_bootstrap_expert_dones or self.model._classifier_output:
                                # forcing expert dones on if classifier output because doesn't make sense otherwise
                                dones[start:end, task_i] = 1.0
                                dones[:start, task_i] = 0.0
                                dones[end:num_expert_total, task_i] = 0.0

                    elif self._reward_model == c.SPARSE:
                        # only going to label non-successful rewards as same as what we would for sqil
                        rews[rews != 1.0] = self.algo_params.get(c.SQIL_POLICY_REWARD_LABEL, 0.0)
                    else:
                        raise NotImplementedError(f"LfGP not implemented for reward_model {self._reward_model}")

                    for task_i in range(self._num_tasks):
                        update_info[f"{c.DISCRIMINATOR_REWARD}/task_{task_i}"].append(rews[:, task_i].cpu().numpy())

                # for n-step, with summing n rewards + discount, only final discount should be used for td update
                if self._sample_horizon > 2 and not self._n_step_mode == 'nth_q_targ':
                    if self._n_step_mode == 'n_rew_only':
                        discounting = full_dis_prod[:, 1]
                    else:
                        discounting = full_dis_prod[:, -1]  # includes fixed gamma as well
                else:
                    if self._sample_horizon > 2:
                        discounting = infos[c.DISCOUNTING][:, 1] * self._gamma
                    else:
                        # note that this used to be done directly in the q update
                        discounting = infos[c.DISCOUNTING] * self._gamma

                update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)

                # sampling and reward updating done, now complete all updates
                for batch_i in range(self._num_prefetch):
                    self._update_num += 1
                    batch_start_idx = batch_i * self._batch_size

                    # note, these are NOT auxiilary tasks in the way we use them for LfGP
                    # this would be some type of auxiliary loss
                    aux_loss, aux_update_info = self._aux_tasks.compute_loss(next_obs, next_h_state)
                    if hasattr(aux_loss, c.BACKWARD):
                        aux_loss.backward()

                    # if we're doing n step and using the nth q value, as is done in RCE, get those samples here
                    if self._sample_horizon > 2 and self._n_step_mode == 'nth_q_targ':
                        # since we set up the trajectory buffer to output t:t+n in all_obss,
                        # all_next_obss has t+1:t+n+1, so the 2nd last all_next_obss will always have t+n
                        # this specifically handles the case where t is the final step before switch/done,
                        # and the second value of all_obs is still just t, even though all_next_obss will have t+1
                        n_obss = all_next_obss[:, -2][:, None, :]
                        n_h_states = all_next_h_states[:, -2][:, None]
                        n_discounts = full_dis_prod[torch.arange(full_dis_prod.shape[0]), ep_lengths][:, None]
                    else:
                        n_obss, n_h_states, n_discounts = None, None, None

                    self.update_qs(batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states,
                                   discounting, infos, lengths, update_info, n_obss, n_h_states, n_discounts,
                                   discount_includes_gamma=True)
                    self._aux_tasks.step()
                    update_info.update(aux_update_info)

                    if self._update_num % self._actor_update_interval == 0:
                        pol_obss, pol_acts, pol_h_states, pol_lengths = obss, acts, h_states, lengths

                        if self._expert_buffer_sampling and self._expert_buffer_model_no_policy:
                            # switch pol obss to only be the ones that we're training q on, to match other implementations
                            pol_obss = obss[self._expert_ends[-1]:]
                            pol_h_states = h_states[self._expert_ends[-1]:]
                            pol_acts = acts[self._expert_ends[-1]:]
                            pol_lengths = lengths[self._expert_ends[-1]:]

                        # ignore absorbing states if we're updating expert policy and not sharing across all tasks,
                        # since in that case we use specific indices of obss, which will break the absorbing states check
                        # in SACDAC's update_policy and update_alpha
                        # also ignore them if they're not in the config
                        ignore_absorbing = (self._expert_buffer_sampling and not \
                            self._expert_buf_policy_share_all and not self._expert_buffer_model_no_policy) or \
                            not self._use_absorbing

                        self.update_policy(batch_start_idx, pol_obss, pol_h_states, pol_acts, rews, dones, next_obss,
                                            next_h_states, discounting, infos, pol_lengths, update_info, ignore_absorbing)

                        # Update Alpha
                        if self.learn_alpha and not isinstance(self.model, MultiTaskFullyConnectedSquashedGaussianTD3):
                            self.update_alpha(batch_start_idx, pol_obss, pol_h_states, pol_acts, rews, dones, next_obss,
                                              next_h_states, discounting, infos, pol_lengths, update_info, ignore_absorbing)

                    if self._update_num % self._target_update_interval == 0:
                        update_info[c.TARGET_UPDATE_TIME] = []
                        tic = timeit.default_timer()
                        self._update_target_network()
                        update_info[c.TARGET_UPDATE_TIME].append(timeit.default_timer() - tic)

                    update_info[c.ALPHA].append(self.model.alpha.detach().cpu().numpy())

            if hasattr(self.model, c.VALUE_RMS):
                update_info[f"{c.VALUE_RMS}/{c.MEAN}"] = self.model.value_rms.mean.numpy()
                update_info[f"{c.VALUE_RMS}/{c.VARIANCE}"] = self.model.value_rms.var.numpy()
            return True, update_info
        return False, update_info


class UpdateSACDACIntentionsPlusHandcraft(UpdateSACDACIntentions):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask()):
        super().__init__(model=model,
                         policy_opt=policy_opt,
                         qs_opt=qs_opt,
                         alpha_opt=alpha_opt,
                         learn_alpha=learn_alpha,
                         buffer=buffer,
                         algo_params=algo_params,
                         aux_tasks=aux_tasks)
        self._num_tasks = algo_params.get(c.NUM_TRAIN_TASKS, 1)