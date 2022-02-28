import torch
import torch.nn as nn

from torch.distributions import Normal

from rl_sandbox.constants import CPU
from rl_sandbox.model_architectures.shared import Flatten
from rl_sandbox.model_architectures.utils import construct_linear_layers, RunningMeanStd


class ActionConditionedFullyConnectedDiscriminator(nn.Module):
    def __init__(self, obs_dim, action_dim, output_dim, layers, device=torch.device(CPU)):
        super().__init__()
        self.device = device

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._output_dim = output_dim

        self._flatten = Flatten()
        self.fc_layers = construct_linear_layers(layers)

        self.output = nn.Linear(layers[-1][1], output_dim)

        self.to(device)

    def forward(self, obss, acts):
        batch_size = obss.shape[0]

        obss = obss.reshape(batch_size, -1)
        x = torch.cat((obss, acts), dim=-1)
        x = self._flatten(x)

        x = x.to(self.device)
        for layer in self.fc_layers:
            x = layer(x)

        logits = self.output(x)

        return logits


class ActionConditionedFullyConnectedDiscriminatorPlusRewards(nn.Module):
    def __init__(self, obs_dim, action_dim, output_dim, handcraft_rewards, layers, device=torch.device(CPU)):
        """
        handcraft_rewards is a dict containing the indices and corresponding reward functions that should be output
        in place of NN outputs.
        """
        super().__init__()
        self.device = device

        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._output_dim = output_dim
        self._handcraft_rewards = handcraft_rewards
        self._true_output_dim = output_dim + len(self._handcraft_rewards)

        # need these so that handcrafted reward magnitudes can be rescaled to match average of learned reward magnitudes
        self._trainable_logits_rms = RunningMeanStd(shape=(self._output_dim,), norm_dim=(0,))
        self._trainable_logits_rms.to(device)
        self._handcraft_rewards_rmss = [RunningMeanStd(shape=(1,), norm_dim=(0,))] * len(self._handcraft_rewards)
        for rms in self._handcraft_rewards_rmss:
            rms.to(device)

        # get trainable indices of true output
        handcraft_indices = [index for index in self._handcraft_rewards.keys()]
        self._trainable_indices = sorted(list(set(range(self._true_output_dim)) ^ set(handcraft_indices)))

        self._flatten = Flatten()
        self.fc_layers = construct_linear_layers(layers)

        self.output = nn.Linear(layers[-1][1], output_dim)

        self.to(device)

    def forward(self, obss, acts):
        batch_size = obss.shape[0]

        obss = obss.reshape(batch_size, -1)
        x = torch.cat((obss, acts), dim=-1)
        x = self._flatten(x)

        x = x.to(self.device)
        for layer in self.fc_layers:
            x = layer(x)

        logits = self.output(x)

        self._trainable_logits_rms.update(logits.detach())
        trainable_mean = self._trainable_logits_rms.mean.mean()
        trainable_var = self._trainable_logits_rms.var.mean()

        full_out = torch.zeros([batch_size, self._true_output_dim]).to(self.device)
        full_out[:, self._trainable_indices] = logits

        obss = obss.to(self.device)
        acts = acts.to(self.device)
        for list_index, (index, func) in enumerate(self._handcraft_rewards.items()):
            rms = self._handcraft_rewards_rmss[list_index]
            rews = func(None, acts, obss, torch_multi=True)
            rms.update(rews)
            rews_normalized = rms.normalize(rews)
            rews_scale_matched = rews_normalized * torch.sqrt(trainable_var + self._trainable_logits_rms.epsilon) + \
                                 trainable_mean
            full_out[:, index] = rews_scale_matched

        return full_out