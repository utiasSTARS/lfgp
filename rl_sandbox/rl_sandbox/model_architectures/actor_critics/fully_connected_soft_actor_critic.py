import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.distributions.utils import _standard_normal

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
                         normalize_value=normalize_value,
                         **kwargs)
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


class FullyConnectedSeparateTD3(FullyConnectedSeparate):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 act_std=1.,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         action_dim=action_dim,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value,
                         **kwargs)
        self._act_std = act_std

        self._policy = nn.Sequential(nn.Linear(obs_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, action_dim))

        self.to(self.device)

    def forward(self, x, h, act_std=None, **kwargs):
        if act_std is None:
            act_std = self._act_std

        x = self._extract_features(x)
        a_mean = self._policy(x)
        a_std = torch.ones_like(a_mean) * act_std

        raise NotImplementedError("Once multitask variant is working, fix this to match that.")

        dist = Normal(a_mean, a_std)
        t_a_mean = self._squash_gaussian(a_mean)
        min_q, _, _ = self._q_vals(x, t_a_mean)
        val = min_q

        return dist, val, h


class FullyConnectedSquashedGaussianSAC(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 shared_layers,
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
                         normalize_value=normalize_value,
                         **kwargs)
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


class Repeat(nn.Module):
    def __init__(self, rep_amt):
        super().__init__()
        self._rep_amt = rep_amt
    def forward(self, x):
        return x.repeat(1, self._rep_amt).unsqueeze(-1)
    def extra_repr(self) -> str:
        return f'rep_amt={self._rep_amt}'


class MultiTaskFullyConnectedSquashedGaussianSAC(SquashedGaussianSoftActorCritic):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 task_dim,
                 shared_layers=None,
                 initial_alpha=1.,
                 eps=1e-7,
                 device=torch.device(CPU),
                 normalize_obs=False,
                 normalize_value=False,
                 branched_outputs=False,
                 task_shared_layers_only=False,
                 num_extra_hidden=0,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         initial_alpha=initial_alpha,
                         eps=eps,
                         norm_dim=(0,),
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=False,
                         **kwargs)
        self._task_dim = task_dim
        self._action_dim = action_dim
        self._flatten = Flatten()
        self._task_shared_layers_only = task_shared_layers_only
        self.branched_outputs = branched_outputs

        num_q_to_add = action_dim

        if shared_layers is None and not task_shared_layers_only:
            self._shared_network = nn.ModuleList([nn.Identity()])
            num_inputs = obs_dim
            if self.branched_outputs:
                extras = [[nn.Conv1d(256 * task_dim, 256 * task_dim, kernel_size=1, groups=task_dim), nn.ReLU()],
                        [nn.Conv1d(256 * task_dim, 256 * task_dim, kernel_size=1, groups=task_dim), nn.ReLU()],
                        [nn.Conv1d(256 * task_dim, 256 * task_dim, kernel_size=1, groups=task_dim), nn.ReLU()]
                ]  # need extras to get 2 hidden layers
            else:
                extras = [[nn.Linear(256, 256), nn.ReLU()],
                          [nn.Linear(256, 256), nn.ReLU()],
                          [nn.Linear(256, 256), nn.ReLU()]
                ]
            task_shared_layers = [[]] * 3
        else:
            if task_shared_layers_only: assert shared_layers is not None, \
                "must define shared_layers if task_shared_layers_only is True"
            num_inputs = shared_layers[-1][1]
            extras = [[]] * 3
            if task_shared_layers_only:
                self._shared_network = nn.ModuleList([nn.Identity()])

                # shared_layers argument is assuming obs_dim only right now
                critic_shared_layers = copy.deepcopy(shared_layers)
                critic_shared_layers[0][0] = shared_layers[0][0] + action_dim
                task_shared_layers = [
                    [*construct_linear_layers(shared_layers)],
                    [*construct_linear_layers(critic_shared_layers)],
                    [*construct_linear_layers(critic_shared_layers)]
                ]
                if branched_outputs:
                    for tsl in task_shared_layers:
                        tsl.append(Repeat(self._task_dim))
                num_q_to_add = 0
            else:
                self._shared_network = construct_linear_layers(shared_layers)
                task_shared_layers = [[]] * 3

        extra_hidden = [[], [], []]
        for _ in range(num_extra_hidden):
            if self.branched_outputs:
                for model_i in range(3):
                    extra_hidden[model_i].extend(
                        [nn.Conv1d(256 * task_dim, 256 * task_dim, kernel_size=1, groups=task_dim), nn.ReLU()])
            else:
                extra_hidden[model_i].extend([nn.Linear(256, 256), nn.ReLU()])

        if self.branched_outputs:
            self._policy = nn.Sequential(
                *task_shared_layers[0],
                nn.Conv1d(num_inputs * task_dim, 256 * task_dim, kernel_size=1, groups=task_dim),
                nn.ReLU(),
                *extras[0],
                *extra_hidden[0],
                nn.Conv1d(256 * task_dim, task_dim * action_dim * 2, kernel_size=1, groups=task_dim))
            self._q1 = nn.Sequential(*task_shared_layers[1],
                                     nn.Conv1d((num_inputs + num_q_to_add) * task_dim, 256 * task_dim,
                                               kernel_size=1, groups=task_dim),
                                     nn.ReLU(),
                                     *extras[1],
                                     *extra_hidden[1],
                                     nn.Conv1d(256 * task_dim, task_dim, kernel_size=1, groups=task_dim))
            self._q2 = nn.Sequential(*task_shared_layers[2],
                                     nn.Conv1d((num_inputs + num_q_to_add) * task_dim, 256 * task_dim,
                                               kernel_size=1, groups=task_dim),
                                     nn.ReLU(),
                                     *extras[2],
                                     *extra_hidden[2],
                                     nn.Conv1d(256 * task_dim, task_dim, kernel_size=1, groups=task_dim))
        else:
            self._policy = nn.Sequential(*task_shared_layers[0],
                                         nn.Linear(num_inputs, 256),
                                         nn.ReLU(),
                                         *extras[0],
                                         *extra_hidden[0],
                                         nn.Linear(256, task_dim * action_dim * 2))
            self._q1 = nn.Sequential(*task_shared_layers[1],
                                     nn.Linear(num_inputs + action_dim, 256),
                                     nn.ReLU(),
                                     *extras[1],
                                     *extra_hidden[1],
                                     nn.Linear(256, task_dim))
            self._q2 = nn.Sequential(*task_shared_layers[2],
                                     nn.Linear(num_inputs + action_dim, 256),
                                     nn.ReLU(),
                                     *extras[2],
                                     *extra_hidden[2],
                                     nn.Linear(256, task_dim))

        self._log_alpha = nn.Parameter(torch.ones(task_dim) * torch.log(torch.tensor(initial_alpha)))

        self.to(self.device)

        if normalize_value:
            self.value_rms = RunningMeanStd(shape=(self._task_dim,), norm_dim=(0,))

        self.count = 0

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
            if not self._task_shared_layers_only:
                # repeat happens within the network list otherwise
                input = input.repeat(1, self._task_dim).unsqueeze(-1)
        q1_val = self._q1(input)
        q2_val = self._q2(input)

        if self._classifier_output:
            q1_val = torch.sigmoid(q1_val)
            q2_val = torch.sigmoid(q2_val)

        if self.branched_outputs:
            q1_val = q1_val.squeeze(-1)
            q2_val = q2_val.squeeze(-1)
        min_q = torch.min(q1_val, q2_val)
        return min_q, q1_val, q2_val

    def forward(self, x, h, **kwargs):
        x = self._extract_features(x)

        if self.branched_outputs:
            if self._task_shared_layers_only:
                x_rep = x  # repeat happens within the network list in this case
            else:
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

class TruncatedNormal(Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class MultiTaskFullyConnectedSquashedGaussianTD3(MultiTaskFullyConnectedSquashedGaussianSAC):
    def __init__(self, obs_dim, action_dim, task_dim, shared_layers, policy_stddev=1.,
                 device=torch.device(CPU), normalize_obs=False, normalize_value=False, branched_outputs=False,
                 squash_last=True, no_squash_act_clip=0.3,
                 **kwargs):
        super().__init__(obs_dim=obs_dim,
                         action_dim=action_dim,
                         task_dim=task_dim,
                         shared_layers=shared_layers,
                         device=device,
                         normalize_obs=normalize_obs,
                         normalize_value=normalize_value,
                         branched_outputs=branched_outputs,
                         **kwargs)
        self._act_std = torch.as_tensor(policy_stddev, device=self.device)
        self._squash_last = squash_last
        self._no_squash_act_clip = no_squash_act_clip

        if self.branched_outputs:
            self._policy = nn.Sequential(
                nn.Conv1d(shared_layers[-1][1] * task_dim, 256 * task_dim, kernel_size=1, groups=task_dim),
                nn.ReLU(),
                nn.Conv1d(256 * task_dim, task_dim * action_dim, kernel_size=1, groups=task_dim))
        else:
            self._policy = nn.Sequential(nn.Linear(shared_layers[-1][1], 256),
                                         nn.ReLU(),
                                         nn.Linear(256, task_dim * action_dim))

        self.to(self.device)

    def forward(self, x, h, return_raw_act=False, **kwargs):
        x = self._extract_features(x)

        if self.branched_outputs:
            x_rep = x.repeat(1, self._task_dim).unsqueeze(-1)
            a_mean = self._policy(x_rep).squeeze(-1)
        else:
            a_mean = self._policy(x)
        a_mean = a_mean.reshape(-1, self._task_dim, self._action_dim)
        a_std = torch.ones_like(a_mean, device=self.device) * self._act_std

        t_a_mean = self._squash_gaussian(a_mean)

        if self._squash_last:
            # forward --> mean + std --> sample --> squash
            dist = Normal(a_mean, a_std)
        else:
            # forward --> squash --> mean + std --> sample + clip  ----- matches drqv2
            dist = TruncatedNormal(t_a_mean, a_std)

        min_q, _, _ = self._q_vals(x, t_a_mean[:, 0])
        val = min_q

        if return_raw_act:
            return dist, val, h, a_mean
        else:
            return dist, val, h

    def _lprob(self, dist, a, t_a):
        if self._squash_last:
            return torch.sum(dist.log_prob(a) - self._squash_gaussian.log_abs_det_jacobian(a, t_a), dim=-1, keepdim=True)
        else:
            return torch.sum(dist.log_prob(a), dim=-1, keepdim=True)

    def act_lprob(self, x, h, clip=None, return_raw_act=False, **kwargs):
        if return_raw_act:
            dist, _, _, raw_mean = self.forward(x, h, return_raw_act=True)
        else:
            dist, _, _ = self.forward(x, h)

        if self._squash_last:
            samp_action = dist.rsample()
            action = self._squash_gaussian(samp_action)
            log_prob = self._lprob(dist, samp_action, action)
        else:
            action = dist.sample(clip=clip)
            log_prob = self._lprob(dist, action)


        if return_raw_act:
            return action, log_prob, raw_mean
        else:
            return action, log_prob

    def deterministic_act_lprob(self, x, h, **kwargs):
        dist, _, _ = self.forward(x, h)

        if self._squash_last:
            act_mean = dist.mean
            action = self._squash_gaussian(act_mean)
            log_prob = self._lprob(dist, act_mean, action)
        else:
            action = dist.mean
            log_prob = self._lprob(dist, action)

        return action, log_prob

    def compute_action(self, x, h, clip=None):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)

            if self._squash_last:
                samp_action = dist.rsample()
                action = self._squash_gaussian(samp_action)
                log_prob = self._lprob(dist, samp_action, action)
            else:
                action = dist.sample(clip=clip)
                log_prob = self._lprob(dist, action)

        self.train()
        return action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy(), dist.mean[0].cpu().numpy(), dist.variance[0].cpu().numpy()

    def deterministic_action(self, x, h):
        self.eval()
        with torch.no_grad():
            dist, value, h = self.forward(x, h=h)

            if self._squash_last:
                act_mean = dist.mean
                action = self._squash_gaussian(act_mean)
                log_prob = self._lprob(dist, act_mean, action)
            else:
                action = dist.mean
                log_prob = self._lprob(dist, action)

        self.train()
        return action[0].cpu().numpy(), value[0].cpu().numpy(), h[0].cpu().numpy(), log_prob[0].cpu().numpy(), dist.entropy()[0].cpu().numpy()


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
