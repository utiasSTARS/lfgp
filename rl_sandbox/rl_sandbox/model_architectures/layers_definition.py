import torch.nn as nn

import rl_sandbox.constants as c
import rl_sandbox.model_architectures.shared as snn


VALUE_BASED_LINEAR_LAYERS = lambda in_dim: (
    [in_dim,  256,      nn.ReLU(), True, 0],
    [256,     256,      nn.ReLU(), True, 0],
)

SAC_DISCRIMINATOR_LINEAR_LAYERS = lambda in_dim: (
    [in_dim,  256,      nn.Tanh(), True, 0],
    [256,     256,      nn.Tanh(), True, 0],
)
