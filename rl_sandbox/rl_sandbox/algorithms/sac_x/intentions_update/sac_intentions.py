import timeit
import torch
import numpy as np

import rl_sandbox.constants as c

from rl_sandbox.algorithms.sac.sac import SAC
from rl_sandbox.algorithms.utils import aug_data
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask

# from robust_loss_pytorch.general import lossfun as robust_loss

class UpdateSACIntentions(SAC):
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
        self._action_dim = algo_params[c.ACTION_DIM]
        self._prev_bias_val = 0
        self._magnitude_penalty_multiplier = self.algo_params.get(c.MAGNITUDE_PENALTY_MULTIPLIER, None)

    def _compute_qs_loss(self, obss, h_states, acts, rews, dones, next_obss, discounting, lengths):
        batch_size = len(obss)
        task_batch_size = batch_size * self._num_tasks

        tasks = torch.arange(self._num_tasks, device=self.device).repeat(batch_size).reshape(task_batch_size, 1)
        rews = rews.reshape(task_batch_size, 1) * self._reward_scaling
        dones = dones.repeat(1, self._num_tasks).reshape(task_batch_size, 1)
        discounting = discounting.repeat(1, self._num_tasks).reshape(task_batch_size, 1)

        rews, dones, discounting = rews.to(self.device), dones.to(self.device), discounting.to(self.device)
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        with torch.no_grad():
            alpha = self.model.alpha.detach().repeat(batch_size).reshape(task_batch_size, 1)

            next_acts, next_lprobs = self.model.act_lprob(next_obss, next_h_states)
            next_acts = next_acts.reshape(task_batch_size, self._action_dim)
            next_lprobs = next_lprobs.reshape(task_batch_size, 1)

            _, _, _, targ_next_h_states = self._target_model.q_vals(obss, h_states, acts, lengths=lengths)

            min_q_targ, _, _, _ = self._target_model.q_vals(
                aug_data(data=next_obss, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
                aug_data(data=targ_next_h_states, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
                next_acts)
            min_q_targ = torch.gather(min_q_targ, dim=1, index=tasks)
            min_q_targ = min_q_targ.detach()

            if hasattr(self.model, c.VALUE_RMS):
                min_q_targ = self.model.value_rms.unnormalize(
                    min_q_targ.reshape(-1, self._num_tasks).cpu()).reshape(task_batch_size, -1).to(self.device)

            v_next = (min_q_targ - alpha * next_lprobs)

            target = rews + (self._gamma ** discounting) * (1 - dones) * v_next

            if hasattr(self.model, c.VALUE_RMS):
                target = target.cpu().reshape(batch_size, self._num_tasks)
                self.model.value_rms.update(target)
                target = self.model.value_rms.normalize(target)
                target = target.to(self.device).reshape(task_batch_size, 1)

        q1_loss = ((q1_val.reshape(task_batch_size, 1) - target) ** 2).sum()
        q2_loss = ((q2_val.reshape(task_batch_size, 1) - target) ** 2).sum()

        return q1_loss, q2_loss

    def _compute_pi_loss(self, obss, h_states, acts, lengths):
        batch_size = len(obss)
        task_batch_size = batch_size * self._num_tasks

        tasks = torch.arange(self._num_tasks, device=self.device).repeat(batch_size).reshape(task_batch_size, 1)

        acts, lprobs = self.model.act_lprob(obss, h_states)
        acts = acts.reshape(task_batch_size, self._action_dim)
        lprobs = lprobs.reshape(task_batch_size, 1)

        min_q, _, _, _ = self.model.q_vals(
            aug_data(data=obss, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
            aug_data(data=h_states, num_aug=self._num_tasks, aug_batch_size=task_batch_size),
            acts,
            lengths=lengths.repeat(1, self._num_tasks).reshape(task_batch_size))
        min_q = torch.gather(min_q, dim=1, index=tasks)

        with torch.no_grad():
            alpha = self.model.alpha.detach().repeat(batch_size).reshape(task_batch_size, 1)
        pi_loss = (alpha * lprobs - min_q).sum()

        return pi_loss

    def _compute_alpha_loss(self, obss, h_states, lengths):
        batch_size = len(obss)
        task_batch_size = batch_size * self._num_tasks
        with torch.no_grad():
            _, lprobs = self.model.act_lprob(obss, h_states, lengths=lengths)
            lprobs = lprobs.reshape(task_batch_size, 1)

        alpha = self.model.alpha.repeat(batch_size).reshape(task_batch_size, 1)
        alpha_loss = (-alpha * (lprobs + self._target_entropy).detach()).sum()

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

            for _ in range(self._num_gradient_updates // self._num_prefetch):
                tic = timeit.default_timer()

                obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths = self.buffer.sample_with_next_obs(
                        self._batch_size * self._num_prefetch, next_obs, next_h_state)

                decay = self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_DECAY, 1.0)
                self._expert_buffer_rate = self._expert_buffer_rate * decay

                if self.algo_params.get(c.EXPERT_BUFFER_MODEL_SHARE_ALL, False) and \
                        int(self._expert_buffer_rate * self._batch_size * self._num_prefetch) > 0:
                    assert self.expert_buffers is not None
                    num_expert_total = int(self._expert_buffer_rate * \
                        self._batch_size * self._num_prefetch)
                    num_policy = int(self._batch_size * self._num_prefetch - num_expert_total)
                    num_per_buffer = int(num_expert_total / len(self.expert_buffers))
                    num_main = num_expert_total - (len(self.expert_buffers) - 1) * num_per_buffer
                    amounts = [num_per_buffer] * len(self.expert_buffers)
                    amounts[self.algo_params['main_intention']] = num_main
                    inds_to_update = np.cumsum(amounts)
                    starts = np.concatenate([[0], inds_to_update[:-1]])
                    ends = inds_to_update
                    for task_i, (start, end, amount) in enumerate(zip(starts, ends, amounts)):
                        e_obss, e_h_states, e_acts, e_rews, e_dones, e_next_obss, e_next_h_states, e_infos, e_lengths = \
                            self.expert_buffers[task_i].sample_with_next_obs(amount, None, None)
                        obss[start:end] = e_obss
                        h_states[start:end] = e_h_states
                        acts[start:end] = e_acts

                        # these might not be the correct reward indices, but they get overwritten anyways so doesn't matter for now
                        # specifically, if the expert buffer reward dimension is diff from the env, which can happen
                        # if we're reusing data but removing one of the intentions, these rewards will be wrong
                        rews[start:end] = e_rews[:, :self.algo_params['num_tasks']]
                        dones[start:end] = e_dones
                        next_obss[start:end] = e_next_obss
                        next_h_states[start:end] = e_next_h_states
                        for k in infos:
                            infos[k][start:end] = e_infos[k]
                        lengths[start:end] = e_lengths

                obss = self.train_preprocessing(obss)
                next_obss = self.train_preprocessing(next_obss)

                if hasattr(self.model, c.OBS_RMS):
                    self.model.obs_rms.update(obss)

                # NOTE: This computes the new reward using the provided reward function (discriminator)
                with torch.no_grad():
                    idxes = lengths.unsqueeze(-1).repeat(1, *obss.shape[2:]).unsqueeze(1)
                    last_obss = torch.gather(obss, axis=1, index=idxes - 1)[:, 0, :]
                    rews = reward_function(last_obss, acts).detach()
                    for task_i in range(self._num_tasks):
                        update_info[f"{c.DISCRIMINATOR_REWARD}/task_{task_i}"].append(rews[:, task_i].cpu().numpy())
                    rews = rews * self._reward_scaling

                discounting = infos[c.DISCOUNTING]
                update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)

                for batch_i in range(self._num_prefetch):
                    self._update_num += 1
                    batch_start_idx = batch_i * self._batch_size
                    # Update Q functions

                    # Auxiliary tasks are usually for shared layers, which is updated along with Q
                    aux_loss, aux_update_info = self._aux_tasks.compute_loss(next_obs, next_h_state)
                    if hasattr(aux_loss, c.BACKWARD):
                        aux_loss.backward()
                    self.update_qs(batch_start_idx,
                                   obss,
                                   h_states,
                                   acts,
                                   rews,
                                   dones,
                                   next_obss,
                                   next_h_states,
                                   discounting,
                                   infos,
                                   lengths,
                                   update_info)
                    self._aux_tasks.step()
                    update_info.update(aux_update_info)

                    if self._update_num % self._actor_update_interval == 0:
                        # Update policy
                        self.update_policy(batch_start_idx,
                                           obss,
                                           h_states,
                                           acts,
                                           rews,
                                           dones,
                                           next_obss,
                                           next_h_states,
                                           discounting,
                                           infos,
                                           lengths,
                                           update_info)

                        # Update Alpha
                        if self.learn_alpha:
                            self.update_alpha(batch_start_idx,
                                              obss,
                                              h_states,
                                              acts,
                                              rews,
                                              dones,
                                              next_obss,
                                              next_h_states,
                                              discounting,
                                              infos,
                                              lengths,
                                              update_info)

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