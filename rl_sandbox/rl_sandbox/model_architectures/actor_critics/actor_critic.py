import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import timeit

from torch.distributions import Normal
from torch.distributions.transforms import TanhTransform

from rl_sandbox.constants import OBS_RMS, CPU
from rl_sandbox.model_architectures.utils import RunningMeanStd


class ActorCritic(nn.Module):
    def __init__(self,
                 obs_dim,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self._obs_dim = obs_dim

        if normalize_obs:
            if isinstance(obs_dim, int):
                obs_dim = (obs_dim,)
            self.obs_rms = RunningMeanStd(shape=obs_dim, norm_dim=norm_dim, device=self.device)

        if normalize_value:
            self.value_rms = RunningMeanStd(shape=(1,), norm_dim=(0,))

    def _extract_features(self, x):
        x = self._flatten(x)

        x = x.to(self.device)  # otherwise breaks with obs rms

        obs, extra_features = x[:, :self._obs_dim], x[:, self._obs_dim:]
        if hasattr(self, OBS_RMS):
            obs = self.obs_rms.normalize(obs)
        x = torch.cat((obs, extra_features), dim=1)
        x = x.to(self.device)
        return x

    def forward(self, x, **kwargs):
        raise NotImplementedError()

    def evaluate_action(self, x, h, a, **kwargs):
        dist, value, _ = self.forward(x, h, **kwargs)
        log_prob = dist.log_prob(a.clone().detach().to(self.device)).sum(dim=-1, keepdim=True)
        return log_prob, value, dist.entropy()

    def compute_action(self, x, h, **kwargs):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        self.train()
        return action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy(), dist.mean[0].cpu().numpy(), dist.variance[0].cpu().numpy()

    def deterministic_action(self, x, h, **kwargs):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.mean
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        self.train()
        return action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy()


class SoftActorCritic(ActorCritic):
    def __init__(self,
                 obs_dim,
                 initial_alpha=1.,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         norm_dim=norm_dim,
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value,
                         **kwargs)
        assert initial_alpha > 0.
        self._log_alpha = nn.Parameter(torch.ones(1) * torch.log(torch.tensor(initial_alpha)))

    def _q_vals(self, x, h, a):
        input = torch.cat((x, a), dim=1)
        q1_val = self._q1(input)
        q2_val = self._q2(input)
        min_q = torch.min(q1_val, q2_val)

        return min_q, q1_val, q2_val, h

    def q_vals(self, x, h, a, **kwargs):
        x = self._extract_features(x)
        a = a.to(self.device)
        return self._q_vals(x, h, a)

    def act_lprob(self, x, h, **kwargs):
        dist, _, _ = self(x, h)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def forward(self, x, h, **kwargs):
        raise NotImplementedError

    @property
    def log_alpha(self):
        return self._log_alpha

    @property
    def alpha(self):
        return torch.exp(self._log_alpha)

    @property
    def policy_parameters(self):
        return self._policy.parameters()

    @property
    def qs_parameters(self):
        return list(self._q1.parameters()) + list(self._q2.parameters())

    @property
    def soft_update_parameters(self):
        return self.qs_parameters


class SquashedGaussianSoftActorCritic(SoftActorCritic):
    def __init__(self,
                 obs_dim,
                 initial_alpha=1.,
                 eps=1e-7,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 classifier_output=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         norm_dim=norm_dim,
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value,
                         **kwargs)
        self._eps = eps
        self._squash_gaussian = TanhTransform()
        self._classifier_output = classifier_output

    def _q_vals(self, x, a):
        input = torch.cat((x, a), dim=1)
        q1_val = self._q1(input)
        q2_val = self._q2(input)

        if self._classifier_output:
            q1_val = torch.sigmoid(q1_val)
            q2_val = torch.sigmoid(q2_val)

        min_q = torch.min(q1_val, q2_val)
        return min_q, q1_val, q2_val

    def _lprob(self, dist, a, t_a):
        return torch.sum(dist.log_prob(a) - self._squash_gaussian.log_abs_det_jacobian(a, t_a), dim=-1, keepdim=True)

    def q_vals(self, x, h, a, **kwargs):
        a = a.to(self.device)
        x = self._extract_features(x)
        min_q, q1_val, q2_val = self._q_vals(x, a)
        return min_q, q1_val, q2_val, h

    def act_lprob(self, x, h, return_raw_act=False, **kwargs):
        dist, _, _ = self.forward(x, h)
        action = dist.rsample()
        t_action = self._squash_gaussian(action)
        log_prob = self._lprob(dist, action, t_action)
        if return_raw_act:
            return t_action, log_prob, dist.mean
        else:
            return t_action, log_prob

    def deterministic_act_lprob(self, x, h, **kwargs):
        dist, _, _ = self.forward(x, h)
        action = dist.mean
        t_action = self._squash_gaussian(action)
        log_prob = self._lprob(dist, action, t_action)
        return t_action, log_prob

    def compute_action(self, x, h):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.rsample()
            t_action = self._squash_gaussian(action)
            log_prob = self._lprob(dist, action, t_action)
        self.train()
        return t_action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy(), dist.mean[0].cpu().numpy(), dist.variance[0].cpu().numpy()

    def deterministic_action(self, x, h):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.mean
            t_action = self._squash_gaussian(action)
            log_prob = self._lprob(dist, action, t_action)
        self.train()
        return t_action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy()

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        a_std = F.softplus(a_raw_std) + self._eps

        dist = Normal(a_mean, a_std)
        t_a_mean = self._squash_gaussian(a_mean)
        min_q, _, _ = self._q_vals(x, t_a_mean)
        val = min_q - self.alpha * self._lprob(dist, a_mean, t_a_mean)

        return dist, val, h


class SquashedGaussianSoftActorCriticPlusHandcraft(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 handcraft_tasks,
                 action_dim,
                 task_dim,
                 obs_dim,
                 initial_alpha=1.,
                 eps=1e-7,
                 norm_dim=(0,),
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim, initial_alpha, eps, norm_dim, device, normalize_obs, normalize_value, **kwargs)
        self._handcraft_tasks = handcraft_tasks
        self._main_task_original_i = self._handcraft_tasks['main_task'][0]
        self._main_task_new_i = self._handcraft_tasks['main_task'][1]
        self._task_dim = task_dim
        self._action_dim = action_dim

        used_indices = [self._main_task_new_i]
        self._handcraft_indices = []
        for task, task_i in self._handcraft_tasks.items():
            if task == 'open_action' or task == 'close_action':
                used_indices.append(task_i)
                self._handcraft_indices.append(task_i)
        self._handcraft_indices = np.array(sorted(self._handcraft_indices))
        self._train_indices = np.array(sorted(list(
            set(range(task_dim + len(self._handcraft_tasks) - 1)) ^ set(self._handcraft_indices))))
        self._non_handcraft_aux_new_indices = np.array(sorted(list(
            set(range(task_dim + len(self._handcraft_tasks) - 1)) ^ set(used_indices))))
        self._non_handcraft_aux_original_indices = np.array(sorted(list(
            set(range(task_dim)) ^ set([self._main_task_original_i]))))

    def action_to_action_with_handcraft(self, dist, action):
        full_action = torch.zeros([1, len(self._handcraft_tasks) - 1 + self._task_dim, self._action_dim]).to(
            self.device)
        for task, task_i in self._handcraft_tasks.items():
            if task == 'open_action':
                main_task_action = copy.deepcopy(action[0, self._main_task_original_i])
                main_task_action[-1] = -1.0
                full_action[0, task_i] = main_task_action
            elif task == 'close_action':
                main_task_action = copy.deepcopy(action[0, self._main_task_original_i])
                main_task_action[-1] = 1.0
                full_action[0, task_i] = main_task_action
            elif task == 'main_task':
                full_action[0, self._main_task_new_i] = action[0, self._main_task_original_i]

        if len(self._non_handcraft_aux_new_indices) > 0:
            full_action[0, self._non_handcraft_aux_new_indices] = action[0, self._non_handcraft_aux_original_indices]

        action = full_action

        t_action = self._squash_gaussian(action)

        new_mean = torch.zeros_like(full_action)
        new_std = torch.zeros_like(full_action)
        new_mean[0, self._handcraft_indices] = dist.mean[0, self._main_task_original_i]
        new_mean[0, self._train_indices] = dist.mean[0]
        new_std[0, self._handcraft_indices] = dist.stddev[0, self._main_task_original_i]
        new_std[0, self._train_indices] = dist.stddev[0]

        new_dist = Normal(new_mean, new_std)
        log_prob = self._lprob(new_dist, action, t_action)

        return t_action, new_dist, log_prob

    def compute_action(self, x, h):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.rsample()

            t_action, dist, log_prob = self.action_to_action_with_handcraft(dist, action)
        self.train()
        return t_action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), \
               dist.entropy()[0].cpu().numpy(), dist.mean[0].cpu().numpy(), dist.variance[0].cpu().numpy()

    def deterministic_action(self, x, h):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)
            action = dist.mean

            t_action, dist, log_prob = self.action_to_action_with_handcraft(dist, action)
        self.train()
        return t_action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy()
