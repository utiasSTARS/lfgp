import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from rl_sandbox.constants import CPU, OBS_RMS
from rl_sandbox.model_architectures.actor_critics.actor_critic import SquashedGaussianSoftActorCritic
from rl_sandbox.model_architectures.actor_critics.actor_critic import SquashedGaussianSoftActorCriticPlusHandcraft
from rl_sandbox.model_architectures.shared import Flatten
from rl_sandbox.model_architectures.utils import construct_linear_layers, RunningMeanStd

class FullyConnectedSeparate(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
        self._action_dim = action_dim
        self._flatten = Flatten()
        self._policy = nn.Sequential(nn.Linear(obs_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(obs_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self._q2 = nn.Sequential(nn.Linear(obs_dim + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self.to(self.device)


class FullyConnectedSquashedGaussianSAC(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 shared_layers,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value)
        self._action_dim = action_dim
        self._flatten = Flatten()

        self._shared_network = construct_linear_layers(shared_layers)
        self._policy = nn.Sequential(nn.Linear(shared_layers[-1][1], 256),
                                     nn.ReLU(),
                                     nn.Linear(256, action_dim * 2))
        self._q1 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        self._q2 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

        self.to(self.device)

    def _extract_features(self, x):
        x = super()._extract_features(x)
        for layer in self._shared_network:
            x = layer(x)
        return x

    @property
    def policy_parameters(self):
        return list(super().policy_parameters)

    @property
    def qs_parameters(self):
        return super().qs_parameters + list(self._shared_network.parameters())


class MultiTaskFullyConnectedSquashedGaussianSAC(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 task_dim,
                 shared_layers,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 branched_outputs=False):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=False)
        self._task_dim = task_dim
        self._action_dim = action_dim
        self._flatten = Flatten()

        self._shared_network = construct_linear_layers(shared_layers)

        self.branched_outputs = branched_outputs
        if self.branched_outputs:
            self._policy = nn.Sequential(
                nn.Conv1d(shared_layers[-1][1] * task_dim, 256 * task_dim, kernel_size=1, groups=task_dim),
                nn.ReLU(),
                nn.Conv1d(256 * task_dim, task_dim * action_dim * 2, kernel_size=1, groups=task_dim))
            self._q1 = nn.Sequential(nn.Conv1d((shared_layers[-1][1] + action_dim) * task_dim, 256 * task_dim,
                                               kernel_size=1, groups=task_dim),
                                     nn.ReLU(),
                                     nn.Conv1d(256 * task_dim, task_dim, kernel_size=1, groups=task_dim))
            self._q2 = nn.Sequential(nn.Conv1d((shared_layers[-1][1] + action_dim) * task_dim, 256 * task_dim,
                                               kernel_size=1, groups=task_dim),
                                     nn.ReLU(),
                                     nn.Conv1d(256 * task_dim, task_dim, kernel_size=1, groups=task_dim))
        else:
            self._policy = nn.Sequential(nn.Linear(shared_layers[-1][1], 256),
                                         nn.ReLU(),
                                         nn.Linear(256, task_dim * action_dim * 2))
            self._q1 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, task_dim))
            self._q2 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, task_dim))
        self._log_alpha = nn.Parameter(torch.ones(task_dim) * torch.log(torch.tensor(initial_alpha)))

        self.to(self.device)

        if normalize_value:
            self.value_rms = RunningMeanStd(shape=(self._task_dim,), norm_dim=(0,))

    def _extract_features(self, x):
        x = super()._extract_features(x)
        for layer in self._shared_network:
            x = layer(x)
        return x

    @property
    def policy_parameters(self):
        return list(super().policy_parameters)

    @property
    def qs_parameters(self):
        return super().qs_parameters + list(self._shared_network.parameters())

    def _q_vals(self, x, a):
        input = torch.cat((x, a), dim=1)
        if self.branched_outputs:
            input = input.repeat(1, self._task_dim).unsqueeze(-1)
        q1_val = self._q1(input)
        q2_val = self._q2(input)
        if self.branched_outputs:
            q1_val = q1_val.squeeze(-1)
            q2_val = q2_val.squeeze(-1)
        min_q = torch.min(q1_val, q2_val)
        return min_q, q1_val, q2_val

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        if self.branched_outputs:
            x_rep = x.repeat(1, self._task_dim).unsqueeze(-1)
            branch_outs_list = torch.chunk(self._policy(x_rep).squeeze(-1), chunks=self._task_dim*2, dim=1)
            a_mean = torch.cat([tens for tens in branch_outs_list[slice(0, self._task_dim*2, 2)]], dim=1)
            a_raw_std = torch.cat([tens for tens in branch_outs_list[slice(1, self._task_dim*2, 2)]], dim=1)
        else:
            a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        a_mean = a_mean.reshape(-1, self._task_dim, self._action_dim)
        a_raw_std = a_raw_std.reshape(-1, self._task_dim, self._action_dim)
        a_std = F.softplus(a_raw_std) + self._eps

        dist = Normal(a_mean, a_std)
        t_a_mean = self._squash_gaussian(a_mean)[:, 0]
        min_q, _, _ = self._q_vals(x, t_a_mean)
        val = min_q - self.alpha[0] * self._lprob(dist, a_mean, t_a_mean)[:, 0]

        return dist, val, h


class MultiTaskFullyConnectedSquashedGaussianSACPlusHandcraft(SquashedGaussianSoftActorCriticPlusHandcraft):
    def __init__(self,
                 handcraft_tasks,
                 obs_dim,
                 action_dim,
                 task_dim,
                 shared_layers,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 branched_outputs=False):

        super().__init__(handcraft_tasks=handcraft_tasks,
                         obs_dim=obs_dim,
                         action_dim=action_dim,
                         task_dim=task_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=False)
        self._task_dim = task_dim
        self._action_dim = action_dim
        self._flatten = Flatten()

        self._shared_network = construct_linear_layers(shared_layers)

        self.branched_outputs = branched_outputs
        if self.branched_outputs:
            self._policy = nn.Sequential(
                nn.Conv1d(shared_layers[-1][1] * task_dim, 256 * task_dim, kernel_size=1, groups=task_dim),
                nn.ReLU(),
                nn.Conv1d(256 * task_dim, task_dim * action_dim * 2, kernel_size=1, groups=task_dim))
            self._q1 = nn.Sequential(nn.Conv1d((shared_layers[-1][1] + action_dim) * task_dim, 256 * task_dim,
                                               kernel_size=1, groups=task_dim),
                                     nn.ReLU(),
                                     nn.Conv1d(256 * task_dim, task_dim, kernel_size=1, groups=task_dim))
            self._q2 = nn.Sequential(nn.Conv1d((shared_layers[-1][1] + action_dim) * task_dim, 256 * task_dim,
                                               kernel_size=1, groups=task_dim),
                                     nn.ReLU(),
                                     nn.Conv1d(256 * task_dim, task_dim, kernel_size=1, groups=task_dim))
        else:
            self._policy = nn.Sequential(nn.Linear(shared_layers[-1][1], 256),
                                         nn.ReLU(),
                                         nn.Linear(256, task_dim * action_dim * 2))
            self._q1 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, task_dim))
            self._q2 = nn.Sequential(nn.Linear(shared_layers[-1][1] + action_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, task_dim))
        self._log_alpha = nn.Parameter(torch.ones(task_dim) * torch.log(torch.tensor(initial_alpha)))

        self.to(self.device)

        if normalize_value:
            self.value_rms = RunningMeanStd(shape=(self._task_dim,), norm_dim=(0,))

    def _extract_features(self, x):
        x = super()._extract_features(x)
        for layer in self._shared_network:
            x = layer(x)
        return x

    @property
    def policy_parameters(self):
        return list(super().policy_parameters)

    @property
    def qs_parameters(self):
        return super().qs_parameters + list(self._shared_network.parameters())

    def _q_vals(self, x, a):
        input = torch.cat((x, a), dim=1)
        if self.branched_outputs:
            input = input.repeat(1, self._task_dim).unsqueeze(-1)
        q1_val = self._q1(input)
        q2_val = self._q2(input)
        if self.branched_outputs:
            q1_val = q1_val.squeeze(-1)
            q2_val = q2_val.squeeze(-1)
        min_q = torch.min(q1_val, q2_val)
        return min_q, q1_val, q2_val

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        if self.branched_outputs:
            x_rep = x.repeat(1, self._task_dim).unsqueeze(-1)
            branch_outs_list = torch.chunk(self._policy(x_rep).squeeze(-1), chunks=self._task_dim*2, dim=1)
            a_mean = torch.cat([tens for tens in branch_outs_list[slice(0, self._task_dim*2, 2)]], dim=1)
            a_raw_std = torch.cat([tens for tens in branch_outs_list[slice(1, self._task_dim*2, 2)]], dim=1)
        else:
            a_mean, a_raw_std = torch.chunk(self._policy(x), chunks=2, dim=1)
        a_mean = a_mean.reshape(-1, self._task_dim, self._action_dim)
        a_raw_std = a_raw_std.reshape(-1, self._task_dim, self._action_dim)
        a_std = F.softplus(a_raw_std) + self._eps

        dist = Normal(a_mean, a_std)
        t_a_mean = self._squash_gaussian(a_mean)[:, 0]
        min_q, _, _ = self._q_vals(x, t_a_mean)
        val = min_q - self.alpha[0] * self._lprob(dist, a_mean, t_a_mean)[:, 0]

        return dist, val, h
