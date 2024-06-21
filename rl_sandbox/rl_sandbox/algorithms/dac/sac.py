import timeit
import torch
import torch.nn as nn

import rl_sandbox.constants as c

from rl_sandbox.algorithms.sac.sac import SAC
from rl_sandbox.auxiliary_tasks.auxiliary_tasks import AuxiliaryTask
from rl_sandbox.envs.wrappers.absorbing_state import AbsorbingStateWrapper
from rl_sandbox.buffers.wrappers.noise_wrapper import NoiseBuffer
from rl_sandbox.model_architectures.utils import RunningMeanStd


class SACDAC(SAC):
    def __init__(self, model, policy_opt, qs_opt, alpha_opt, learn_alpha, buffer, algo_params, aux_tasks=AuxiliaryTask(),
                 expert_buffer=None):
        super().__init__(model=model,
                         policy_opt=policy_opt,
                         qs_opt=qs_opt,
                         alpha_opt=alpha_opt,
                         learn_alpha=learn_alpha,
                         buffer=buffer,
                         algo_params=algo_params,
                         aux_tasks=aux_tasks)

        self._action_dim = algo_params[c.ACTION_DIM]

        # expert buffer
        self.expert_buffer = expert_buffer
        self._expert_buffer_rate = self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_RATE, 0.)
        self._sample_horizon = algo_params.get(c.N_STEP, 1) + 1
        self._reward_model = self.algo_params.get(c.REWARD_MODEL, "discriminator")
        self._buffer_sample_batch_size = self._batch_size
        self._expert_end = 0
        self._expert_buffer_sampling = False
        self._expert_buffer_model_no_policy = self.algo_params.get(c.EXPERT_BUFFER_MODEL_NO_POLICY, False)
        self._num_tasks = 1

        # n step
        self._n_step_mode = self.algo_params.get(c.N_STEP_MODE, "")

        # other options
        self._rce_eps = algo_params.get(c.RCE_EPS, c.DEFAULT_RCE_PARAMS[c.EPS])
        self._magnitude_penalty_multiplier = self.algo_params.get(c.MAGNITUDE_PENALTY_MULTIPLIER, None)
        self._no_entropy_in_qloss = self.algo_params.get(c.NO_ENTROPY_IN_QLOSS, False)

        if self._sample_horizon > 2:
            gamma_horizon = [1.0]
            for _ in range(self._sample_horizon - 1):
                gamma_horizon.append(gamma_horizon[-1] * self._gamma)
            self._gamma_horizon = torch.as_tensor(gamma_horizon, device=self.device)
            self._gamma_horizon = self._gamma_horizon.repeat(self._batch_size, 1)

        # checking if we're using absorbing
        self._use_absorbing = False
        for wrap in self.algo_params[c.ENV_SETTING][c.ENV_WRAPPERS]:
            if wrap['wrapper'] == AbsorbingStateWrapper:
                self._use_absorbing = True

        # check if we should randomize next obs when we don't have it
        if self.algo_params.get(c.EXPERT_DATA_MODE, 'obs_act') == 'obs_only_no_next':
            self._obs_only_no_next_add_noise = False
            for wrap in self.algo_params[c.EXPERT_BUFFER_SETTING][c.BUFFER_WRAPPERS]:
                if wrap['wrapper'] == NoiseBuffer:
                    self._obs_only_no_next_add_noise = True

        # handle expbuf_size_type argument
        self._expert_buffer_size_type = self.algo_params.get(c.EXPERT_BUFFER_SIZE_TYPE, "fraction")
        if self._expert_buffer_size_type == "match_bs" and self._expert_buffer_rate > 0.0:
            # now batch sizes are bigger since we append equal amounts of expert data per task to buffer samples
            self._batch_size = self._batch_size * 2  # 1 for expert, 1 for non-expert
            self._num_samples_per_accum = self._batch_size // self._accum_num_grad

    def _compute_qs_loss(self, obss, h_states, acts, rews, dones, next_obss, discounting, lengths, update_info,
                         n_obss=None, n_h_states=None, n_discounts=None, discount_includes_gamma=False):
        rews, dones, discounting = rews.to(self.device), dones.to(self.device), discounting.to(self.device)
        _, q1_val, q2_val, next_h_states = self.model.q_vals(obss, h_states, acts, lengths=lengths)

        if self._expert_end > 0:
            max_expert_q = q1_val[:self._expert_end].max().detach()
            avg_expert_q = q1_val[:self._expert_end].mean().detach()
            update_info[c.MAX_EXPERT_Q].append(max_expert_q.cpu().numpy())
            update_info[c.AVG_EXPERT_Q].append(avg_expert_q.cpu().numpy())

        update_info[c.MAX_POLICY_Q].append(q1_val[self._expert_end:].max().detach().cpu().numpy())
        update_info[c.AVG_POLICY_Q].append(q1_val[self._expert_end:].mean().detach().cpu().numpy())

        # for debugging
        num_latest_done = 5
        latest_dones = self.buffer.dones.nonzero()[-num_latest_done:, 0]
        if len(latest_dones) >= num_latest_done:
            with torch.no_grad():
                _, q_latest_dones, _, _ = self.model.q_vals(
                    self.buffer.observations[latest_dones][:, None, :], h_states[:num_latest_done],
                    self.buffer.actions[latest_dones], lengths=1)
                update_info[c.AVG_DONE_Q].append(q_latest_dones.mean().detach().cpu().numpy())

        # if we want to bootstrap even if "done", set dones to zero
        if self._bootstrap_on_done:
            # if we're using rce/sqil with boot turned off, keep expert dones as is
            if self._reward_model in ['sqil', 'rce'] and \
                    self.algo_params.get(c.SQIL_RCE_BOOTSTRAP_EXPERT_MODE, "no_boot") == "no_boot":
                dones[self._expert_end:] = 0.0
            else:
                dones[:] = 0.0

        rews *= self._reward_scaling
        weights = torch.ones_like(q1_val)
        weights[:self._expert_end] = self.algo_params.get(c.EXPERT_CRITIC_WEIGHT, 1.0)

        with torch.no_grad():
            v_next = self._calc_v_next(h_states, next_obss, next_h_states, n_obss, n_h_states, n_discounts)

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
                weights[self._expert_end:] = 1 + discount[self._expert_end:] * w[self._expert_end:]
                td_targets = td_targets * (1 - dones)

                # now set expert indices to ones, all others are just the targets above

                # this line is irrelevant if you're using a classifier output, so removing entirely
                # particularly to deal with expert datasets that may incorrectly not have expert dones labelled as 1
                # td_targets[:self._expert_end] = 1.0 + (1 - dones[:self._expert_end]) * td_targets[:self._expert_end]

                td_targets[:self._expert_end] = 1.0
                target = td_targets

                # keeping here even though it's set by param now, just in case there are floating point problems
                weights[:self._expert_end] = 1 - discount[:self._expert_end]
            else:
                target = rews + discount * (1 - dones) * v_next

                if self.algo_params.get(c.Q_EXPERT_TARGET_MODE, 'bootstrap') == 'max' and \
                        self.algo_params.get(c.SQIL_RCE_BOOTSTRAP_EXPERT_MODE, "no_boot") == "boot":
                    target[:self._expert_end] = self._reward_scaling * torch.ones_like(target[:self._expert_end]) / \
                        (1 - discount[:self._expert_end])

            # applies to both SQIL and RCE with classifier output
            if self.model._classifier_output:
                td_targets[:self._expert_end] = 1.0

            if hasattr(self.model, c.VALUE_RMS):
                target = target.cpu()
                self.model.value_rms.update(target)
                target = self.model.value_rms.normalize(target).to(self.device)

        if self.model._classifier_output:
            assert self.algo_params.get(c.SQIL_RCE_BOOTSTRAP_EXPERT_MODE, "no_boot") == "no_boot", \
                "Cannot bootstrap on expert done if using classifier output."
            classify_loss = torch.nn.BCELoss(weight=weights, reduction="sum")
            q1_loss = classify_loss(q1_val, target)
            q2_loss = classify_loss(q2_val, target)
        else:
            q1_loss = ((weights * (q1_val - target)) ** 2).sum()
            q2_loss = ((weights * (q2_val - target)) ** 2).sum()

        if self.algo_params.get(c.Q_OVER_MAX_PENALTY, 0.0) > 0:
            penalty = self.algo_params[c.Q_OVER_MAX_PENALTY]
            num_med_filt = self.algo_params.get(c.QOMP_NUM_MED_FILT, 50)

            if self._reward_model == c.SQIL or self._reward_model == c.SPARSE:
                q_max = self._reward_scaling * torch.ones_like(q1_val) / (1 - discount)
                q_min = self._reward_scaling * torch.ones_like(q1_val) / (1 - discount) * \
                    self.algo_params.get(c.SQIL_POLICY_REWARD_LABEL, 0.0)

            elif self._reward_model == c.DISCRIMINATOR:
                min_r = torch.minimum(torch.tensor(-.01), rews.min())
                max_r = torch.maximum(torch.tensor(.01), rews.max())

                if not hasattr(self, "_prev_maxs"):
                    self._prev_maxs = torch.ones(num_med_filt, device=self.device) * max_r
                if not hasattr(self, "_prev_mins"):
                    self._prev_mins = torch.ones(num_med_filt, device=self.device) * min_r

                self._prev_mins = self._prev_mins.roll(1, dims=0)
                self._prev_mins[0] = min_r
                self._prev_maxs = self._prev_maxs.roll(1, dims=0)
                self._prev_maxs[0] = max_r

                min_r_filtered = self._prev_mins.median(axis=0)[0]
                max_r_filtered = self._prev_maxs.median(axis=0)[0]
                q_min = min_r_filtered * torch.ones_like(q1_val) / (1 - discount)
                q_max = max_r_filtered * torch.ones_like(q1_val) / (1 - discount)

            else:
                raise NotImplementedError(f"Q over max penalty not implemented for reward model {self._reward_model}")

            # also set q max for policy to be even more restrictive, based on whatever the current q max for expert is
            if self._expert_end > 0:
                max_q = max_expert_q if self.algo_params.get(c.QOMP_POLICY_MAX_TYPE, 'max_exp') == 'max_exp' else avg_expert_q
                if not hasattr(self, "_prev_q_maxs"):
                    self._prev_q_maxs = torch.ones(num_med_filt, device=self.device) * max_q
                self._prev_q_maxs = self._prev_q_maxs.roll(1, dims=0)
                self._prev_q_maxs[0] = max_q
                max_exp_q_filtered = self._prev_q_maxs.median(axis=0)[0]
                q_max[self._expert_end:, :] = max_exp_q_filtered

            q1_max_mag_loss = penalty * torch.maximum(q1_val - q_max, torch.tensor(0)) ** 2
            q2_max_mag_loss = penalty * torch.maximum(q2_val - q_max, torch.tensor(0)) ** 2
            q1_min_mag_loss = penalty * torch.maximum(-(q1_val - q_min), torch.tensor(0)) ** 2
            q2_min_mag_loss = penalty * torch.maximum(-(q2_val - q_min), torch.tensor(0)) ** 2
            q1_loss = q1_loss + q1_max_mag_loss.sum() + q1_min_mag_loss.sum()
            q2_loss = q2_loss + q2_max_mag_loss.sum() + q2_min_mag_loss.sum()

        if self.algo_params.get(c.NOISE_ZERO_TARGET_MODE, 'none') != 'none':
            obs_min = self.buffer.observations.min(axis=0)[0]
            obs_max = self.buffer.observations.max(axis=0)[0]
            obs_range = obs_max - obs_min
            if self.algo_params[c.NOISE_ZERO_TARGET_MODE] == 'range':
                random_obss_q1 = obs_range * torch.rand(obss.shape, device=self.device) + obs_min
                random_obss_q2 = obs_range * torch.rand(obss.shape, device=self.device) + obs_min
            elif self.algo_params[c.NOISE_ZERO_TARGET_MODE] == 'per_obs':
                scale_param = self.algo_params[c.NZT_PER_OBS_SCALE]
                random_obss_q1 = obss[self._expert_end:] + \
                    obs_range * scale_param * torch.rand(obss[self._expert_end:].shape, device=self.device)
                random_obss_q2 = obss[self._expert_end:] + \
                    obs_range * scale_param * torch.rand(obss[self._expert_end:].shape, device=self.device)
            else:
                raise NotImplementedError()

            # always do fully uniform random actions, regardless of mode
            act_min = -1
            act_max = 1
            random_acts_q1 = (act_max - act_min) * torch.rand([random_obss_q1.shape[0], *acts.shape[1:]], device=self.device) + act_min
            random_acts_q2 = (act_max - act_min) * torch.rand([random_obss_q2.shape[0], *acts.shape[1:]], device=self.device) + act_min

            _, q1_rval, _, next_h_states = self.model.q_vals(random_obss_q1, h_states, random_acts_q1, lengths=lengths)
            _, _, q2_rval, next_h_states = self.model.q_vals(random_obss_q2, h_states, random_acts_q2, lengths=lengths)

            q1_random_loss = (q1_rval ** 2).sum()
            q2_random_loss = (q2_rval ** 2).sum()

            q1_loss += q1_random_loss
            q2_loss += q2_random_loss

        return q1_loss, q2_loss

    def update_policy(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, discounting, infos, lengths, update_info,
                      ignore_absorbing=False):
        tic = timeit.default_timer()
        batch_size = obss.shape[0]
        num_samples_per_accum = batch_size // self._accum_num_grad
        self.policy_opt.zero_grad()
        total_pi_loss = 0.
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * num_samples_per_accum)
            if ignore_absorbing:
                non_absorbing_idx = torch.arange(obss[opt_idxes].shape[0])
            else:
                non_absorbing_idx = torch.where(obss[opt_idxes, -1, -1] == 0)[0]
            pi_loss = self._compute_pi_loss(obss[opt_idxes][non_absorbing_idx],
                                            h_states[opt_idxes][non_absorbing_idx],
                                            acts[opt_idxes][non_absorbing_idx],
                                            lengths[opt_idxes][non_absorbing_idx])
            pi_loss /= len(non_absorbing_idx)
            total_pi_loss += pi_loss.detach().cpu()
            pi_loss.backward()
        nn.utils.clip_grad_norm_(self.model.policy_parameters,
                                self._max_grad_norm)
        self.policy_opt.step()
        update_info[c.POLICY_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.PI_LOSS].append(total_pi_loss.numpy())

    def update_alpha(self, batch_start_idx, obss, h_states, acts, rews, dones, next_obss, next_h_states, discounting, infos, lengths, update_info,
                     ignore_absorbing=False):
        tic = timeit.default_timer()
        batch_size = obss.shape[0]
        num_samples_per_accum = batch_size // self._accum_num_grad
        self.alpha_opt.zero_grad()
        total_alpha_loss = 0.
        for grad_i in range(self._accum_num_grad):
            opt_idxes = range(batch_start_idx + grad_i * num_samples_per_accum,
                              batch_start_idx + (grad_i + 1) * num_samples_per_accum)
            if ignore_absorbing:
                non_absorbing_idx = torch.arange(obss[opt_idxes].shape[0])
            else:
                non_absorbing_idx = torch.where(obss[opt_idxes, -1, -1] == 0)[0]
            alpha_loss = self._compute_alpha_loss(obss[opt_idxes][non_absorbing_idx],
                                                  h_states[opt_idxes][non_absorbing_idx],
                                                  lengths[opt_idxes][non_absorbing_idx])
            alpha_loss /= len(non_absorbing_idx)
            total_alpha_loss += alpha_loss.detach().cpu()
            alpha_loss.backward()
        nn.utils.clip_grad_norm_(self.model.log_alpha,
                                self._max_grad_norm)
        self.alpha_opt.step()
        update_info[c.ALPHA_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.ALPHA_LOSS].append(total_alpha_loss.numpy())

    def update(self, reward_function, next_obs, next_h_state):
        self.step += 1

        update_info = {}

        # Perform SAC update
        if self.step >= self._buffer_warmup and self.step % self._steps_between_update == 0:
            update_info[c.AVG_EXPERT_Q] = []
            update_info[c.MAX_EXPERT_Q] = []
            update_info[c.AVG_POLICY_Q] = []
            update_info[c.AVG_DONE_Q] = []
            update_info[c.MAX_POLICY_Q] = []
            update_info[c.PI_LOSS] = []
            update_info[c.Q1_LOSS] = []
            update_info[c.Q2_LOSS] = []
            update_info[c.ALPHA] = []
            update_info[c.SAMPLE_TIME] = []
            update_info[c.Q_UPDATE_TIME] = []
            update_info[c.POLICY_UPDATE_TIME] = []
            update_info[c.ALPHA_LOSS] = []
            update_info[c.ALPHA_UPDATE_TIME] = []
            update_info[c.DISCRIMINATOR_REWARD] = []
            update_info[c.DISCRIMINATOR_MAX] = []
            update_info[c.DISCRIMINATOR_MIN] = []
            update_info["explore_bonus_scaler"] = []
            update_info["expert_reward"] = []

            self._expert_buffer_sampling = False

            for _ in range(self._num_gradient_updates // self._num_prefetch):
                tic = timeit.default_timer()

                # replace obss/acts with expert data if expert_buffer_rate is set
                decay = self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_DECAY, 1.0)
                self._expert_buffer_rate = self._expert_buffer_rate * decay

                policy_batch_size = self._buffer_sample_batch_size * self._num_prefetch - \
                    int(self._expert_buffer_rate * self._batch_size * self._num_prefetch)

                # sample observations from n step buffer or regular buffer
                # immediately sample the correct number, instead of sampling and overwriting with expert
                if self._sample_horizon > 2:
                    obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, sample_idxes, ep_lengths =\
                        self.buffer.sample_trajs(policy_batch_size, None, None)

                else:
                    obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, sample_idxes = \
                        self.buffer.sample_with_next_obs(policy_batch_size, next_obs, next_h_state,
                            include_random_idxes=True)

                if int(self._expert_buffer_rate * self._batch_size * self._num_prefetch) > 0 or \
                    self._reward_model in ['sqil', 'rce']:
                    ##### START expert buffer sampling

                    assert self.expert_buffer is not None
                    self._expert_buffer_sampling = True
                    amount = int(self._expert_buffer_rate * self._batch_size * self._num_prefetch)

                    # to match old overwrite method, immediately initially populate new data with zeros, to be overwritten
                    obss = torch.cat([torch.zeros([amount, *obss.shape[1:]], device=self.device, dtype=obss.dtype), obss])
                    h_states = torch.cat([torch.zeros([amount, *h_states.shape[1:]], device=self.device, dtype=h_states.dtype), h_states])
                    acts = torch.cat([torch.zeros([amount, *acts.shape[1:]], device=self.device, dtype=acts.dtype), acts])
                    rews = torch.cat([torch.zeros([amount, *rews.shape[1:]], device=self.device, dtype=rews.dtype), rews])
                    dones = torch.cat([torch.zeros([amount, *dones.shape[1:]], device=self.device, dtype=dones.dtype), dones])
                    next_obss = torch.cat([torch.zeros([amount, *next_obss.shape[1:]], device=self.device, dtype=next_obss.dtype), next_obss])
                    next_h_states = torch.cat([torch.zeros([amount, *next_h_states.shape[1:]], device=self.device, dtype=next_h_states.dtype), next_h_states])
                    for k in infos:
                        infos[k] = torch.cat([torch.zeros([amount, *infos[k].shape[1:]], device=self.device, dtype=infos[k].dtype), infos[k]])
                    lengths = torch.cat([torch.zeros([amount, *lengths.shape[1:]], device=self.device, dtype=lengths.dtype), lengths])
                    sample_idxes = torch.cat([torch.zeros([amount, *sample_idxes.shape[1:]], device=self.device, dtype=sample_idxes.dtype), sample_idxes])
                    if self._sample_horizon > 2:
                        ep_lengths = torch.cat([torch.zeros([amount, *ep_lengths.shape[1:]], device=self.device, dtype=ep_lengths.dtype), ep_lengths])

                    start = 0

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
                        amount = self._buffer_sample_batch_size * self._num_tasks

                    end = amount
                    self._expert_end = amount

                    if self._sample_horizon > 2:
                        e_obss, e_h_states, e_acts, e_rews, e_dones, e_next_obss, e_next_h_states, e_infos, \
                        e_lengths, e_sample_idxes, e_ep_lengths = \
                            self.expert_buffer.sample_trajs(amount, None, None)

                        ep_lengths[start:end] = e_ep_lengths
                        rews[start:end] = e_rews[:, :, 0:1] # wrong reward if multitask, but doesn't matter because overwritten
                    else:
                        e_obss, e_h_states, e_acts, e_rews, e_dones, e_next_obss, e_next_h_states, e_infos, e_lengths, \
                            e_sample_idxes = \
                            self.expert_buffer.sample_with_next_obs(amount, None, None, include_random_idxes=True)

                        rews[start:end] = e_rews[:, 0:1] # wrong reward if multitask, but doesn't matter because overwritten

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
                                self.expert_buffer.sample_with_next_obs(
                                    amount, None, None, idxes=e_sample_idxes)
                            next_obss[start:end] = e_obss_resampled
                        else:
                            next_obss[start:end] = obss[start:end]
                        next_h_states[start:end] = h_states[start:end]

                    # replace actions if we're doing obs only expert data
                    all_exp_inds_sl = slice(start, end)
                    if 'obs_only' in self.algo_params.get(c.EXPERT_DATA_MODE, 'obs_act'):
                        # overwrite actions with mean policy output instead...RCE uses samples from policy
                        # we'll try samples first, and then later with mean
                        # see https://github.com/google-research/google-research/blob/ccf75222d6c63dc16a4e195d2b2223d8e1a160b4/rce/rce_agent.py#L547
                        # jan24/2024: this is turning out to be a source of seriously stagnated learning...going to test other options

                        if self._sample_horizon > 2:
                            exp_obs, exp_h_states = obss[all_exp_inds_sl], h_states[all_exp_inds_sl]

                            # pass all n steps through net simultaneously
                            obs_flat_along_n = exp_obs.reshape(-1, exp_obs.shape[-1])
                            h_states_flat_along_n = exp_obs.reshape(-1, exp_h_states.shape[-1])
                            with torch.no_grad():
                                cur_pol_exp_obs_acts_flat, _ = self.model.act_lprob(obs_flat_along_n,
                                                                                    h_states_flat_along_n)

                            cur_pol_exp_obs_acts = cur_pol_exp_obs_acts_flat.reshape(
                                exp_obs.shape[0], exp_obs.shape[1], cur_pol_exp_obs_acts_flat.shape[1])
                            acts[start:end] = cur_pol_exp_obs_acts

                        else:
                            with torch.no_grad():
                                cur_pol_exp_obs_acts, _ = self.model.act_lprob(obss[all_exp_inds_sl],
                                                                                h_states[all_exp_inds_sl])
                            acts[start:end] = cur_pol_exp_obs_acts

                    always_sample_exploration = False
                    if always_sample_exploration:
                    # always sample some amount f   rom the random exploratory data to try and make q smoother
                        assert self.buffer._pointer == self.buffer._count, \
                            "Exploratory data is being overwritten in this case, implement a separate explore buffer if you"\
                            " want to allow memory size to be smaller than max steps"

                        explore_amount_total = min(len(self.buffer), self.algo_params[c.EXPLORATION_STEPS])
                        r_end = end
                        end = round(end / 2)
                        self._expert_end = end
                        r_start = end

                        explore_batch_size = r_end - end
                        r_sample_idxes = torch.randint(low=0, high=explore_amount_total, size=(explore_batch_size,),
                                                    device=self.device)

                        if self._sample_horizon > 2:
                            r_obss, r_h_states, r_acts, r_rews, r_dones, r_next_obss, r_next_h_states, r_infos, r_lengths, _, r_ep_lengths =\
                                self.buffer.sample_trajs(explore_batch_size, None, idxes=r_sample_idxes)
                            ep_lengths[r_start:r_end] = r_ep_lengths
                            rews[r_start:r_end] = r_rews[:, :, 0:1] # wrong reward if multitask, but doesn't matter because overwritten
                        else:
                            r_obss, r_h_states, r_acts, r_rews, r_dones, r_next_obss, r_next_h_states, r_infos, r_lengths = \
                                    self.buffer.sample_with_next_obs(explore_batch_size, None, None, idxes=r_sample_idxes)
                            rews[r_start:r_end] = r_rews[:, 0:1] # wrong reward if multitask, but doesn't matter because overwritten

                        obss[r_start:r_end] = r_obss
                        h_states[r_start:r_end] = r_h_states
                        acts[r_start:r_end] = r_acts
                        dones[r_start:r_end] = r_dones
                        next_obss[r_start:r_end] = r_next_obss
                        next_h_states[r_start:r_end] = r_next_h_states
                        for k in infos:
                            if k in r_infos.keys():
                                infos[k][r_start:r_end] = r_infos[k]
                        lengths[r_start:r_end] = r_lengths
                        sample_idxes[r_start:r_end] = r_sample_idxes

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

                # compute the new reward using the provided reward function (discriminator or otherwise)
                with torch.no_grad():
                    if self._reward_model == 'discriminator':
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
                            raw_rews = reward_function(last_obss, acts).detach()

                            if self.algo_params.get(c.REW_MIN_ZERO, False):
                                min_r = torch.minimum(torch.tensor(-.01), raw_rews.min())

                                if not hasattr(self, "_prev_rew_mins"):
                                    num_med_filt = self.algo_params.get(c.RMZ_NUM_MED_FILT, 50)
                                    self._prev_rew_mins = torch.ones(num_med_filt, device=self.device) * min_r

                                self._prev_rew_mins = self._prev_rew_mins.roll(1, dims=0)
                                self._prev_rew_mins[0] = min_r

                                filtered_min_r = self._prev_rew_mins.median()
                                rews = raw_rews - filtered_min_r

                            else:
                                rews = raw_rews

                    elif self._reward_model == 'sqil' or self._reward_model == 'rce':
                        # the way SQIL is implemented in the RCE paper, the actual expert targets are directly labelled
                        # with 1.0, with no bootstrapping, i.e. as if the expert data is trained with done

                        # for RCE, we'll just use the same logic for dones since we end up ignoring the rewards anyways
                        assert self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_DECAY, 1.0) == 1.0, \
                            "expert buffer sampling must have no decay for sqil"
                        # assert self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_RATE, 0.) == 0.5, \
                        #     "Sample rate of 0.5 matches SQIL"
                        assert self.algo_params.get(c.EXPERT_BUFFER_MODEL_SAMPLE_RATE, 0.) > 0.0, \
                            "Expert sample rate must be above 0 for SQIL/RCE"

                        if self._sample_horizon > 2:
                            assert self._n_step_mode == 'nth_q_targ', "use nth_q_targ to match RCE implementation if using nstep"
                            rews = rews[:, 0]  # since we still have n-step sampled rewards

                        rews[start:end] = 1.0
                        rews[end:] = self.algo_params.get(c.SQIL_POLICY_REWARD_LABEL, 0.0)

                    elif self._reward_model == c.SPARSE:
                        # only going to label non-successful rewards as same as what we would for sqil
                        rews[rews != 1.0] = self.algo_params.get(c.SQIL_POLICY_REWARD_LABEL, 0.0)

                    else:
                        raise NotImplementedError(f"SAC/DAC not implemented for reward_model {self._reward_model}")

                    update_info[f"{c.DISCRIMINATOR_REWARD}"].append(rews[self._expert_end:, 0].cpu().numpy())
                    update_info[f"{c.DISCRIMINATOR_MAX}"].append(rews[self._expert_end:, 0].max().cpu().numpy())
                    update_info[f"{c.DISCRIMINATOR_MIN}"].append(rews[self._expert_end:, 0].min().cpu().numpy())

                # for n-step, with summing n rewards + discount, only final discount should be used for td update
                if self._sample_horizon > 2 and not self._n_step_mode == 'nth_q_targ':
                    if self._n_step_mode == 'n_rew_only':
                        discounting = full_dis_prod[:, 1][:, None]
                    else:
                        discounting = full_dis_prod[:, -1][:, None]  # includes fixed gamma as well
                else:
                    if self._sample_horizon > 2:
                        discounting = infos[c.DISCOUNTING][:, 1][:, None] * self._gamma
                    else:
                        # note that this used to be done directly in the q update
                        discounting = infos[c.DISCOUNTING] * self._gamma

                update_info[c.SAMPLE_TIME].append(timeit.default_timer() - tic)

                # sampling and reward updating done, now complete all updates
                for batch_i in range(self._num_prefetch):
                    self._update_num += 1
                    batch_start_idx = batch_i * self._batch_size
                    # Update Q functions

                    # Auxiliary tasks are usually for shared layers, which is updated along with Q
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
                            pol_obss = obss[self._expert_end:]
                            pol_h_states = h_states[self._expert_end:]
                            pol_acts = acts[self._expert_end:]
                            pol_lengths = lengths[self._expert_end:]

                        # Update policy
                        self.update_policy(batch_start_idx, pol_obss, pol_h_states, pol_acts, rews, dones, next_obss,
                                           next_h_states, discounting, infos, pol_lengths, update_info,
                                           ignore_absorbing=not self._use_absorbing)

                        # Update Alpha
                        if self.learn_alpha:
                            self.update_alpha(batch_start_idx, pol_obss, pol_h_states, pol_acts, rews, dones, next_obss,
                                              next_h_states, discounting, infos, pol_lengths, update_info,
                                              ignore_absorbing=not self._use_absorbing)

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
