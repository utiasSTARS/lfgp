import numpy as np
import torch
import torch.nn as nn

import rl_sandbox.constants as c

def default_weight_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.LSTM or type(m) == nn.GRU:
        torch.nn.init.xavier_uniform_(m.weight_ih_l0)
        torch.nn.init.orthogonal_(m.weight_hh_l0)
        if m.bias is not None:
            m.bias_ih_l0.data.fill_(0)
            m.bias_hh_l0.data.fill_(0)


def construct_linear_layers(layers):
    linear_layers = nn.ModuleList()
    for (in_dim, out_dim, activation, use_bias, dropout_p) in layers:
        linear_layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
        linear_layers.append(activation)
        if dropout_p > 0.:
            linear_layers.append(nn.Dropout(dropout_p))

    return linear_layers


def make_model(model_cfg):
    return model_cfg[c.MODEL_ARCHITECTURE](**model_cfg[c.KWARGS])


def make_optimizer(parameters, optimizer_cfg):
    return optimizer_cfg[c.OPTIMIZER](parameters, **optimizer_cfg[c.KWARGS])


class RunningMeanStd():
    """ Modified from Baseline
    Assumes shape to be (number of inputs, input_shape)
    """

    def __init__(self, epsilon=1e-4, shape=(), norm_dim=(0,), a_min=-5., a_max=5., device='cpu'):
        assert epsilon > 0.
        self.shape = shape
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float)
        self.var = torch.ones(shape, dtype=torch.float)
        self.epsilon = epsilon
        self.count = epsilon
        self.a_min = a_min
        self.a_max = a_max
        self.norm_dim = norm_dim
        self.to(self.device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        eps = torch.tensor([self.epsilon])
        self.epsilon = eps.to(device)

    def update(self, x):
        batch_mean = torch.mean(x, dim=self.norm_dim)
        batch_var = torch.var(x, dim=self.norm_dim)
        batch_count = int(torch.prod(torch.tensor(
            [x.shape[dim] for dim in self.norm_dim])))
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, x):
        x_shape = x.shape
        x = x.reshape(-1, *self.shape).to(self.device)
        normalized_x = torch.clamp((x - self.mean) / torch.sqrt(self.var + self.epsilon),
                                   min=self.a_min,
                                   max=self.a_max)
        normalized_x[normalized_x != normalized_x] = 0.
        normalized_x = normalized_x.reshape(x_shape)
        return normalized_x

    def unnormalize(self, x):
        return x * torch.sqrt(self.var + self.epsilon) + self.mean
