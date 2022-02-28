import numpy as np
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Split(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.feature_dims = feature_dims

    def forward(self, x):
        features = []
        last_feature_idx = 0
        for feature_dim in self.feature_dims:
            features.append(x[..., last_feature_idx:last_feature_idx + feature_dim])
            last_feature_idx += feature_dim
        return features


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * nn.functional.sigmoid(x)
        return x


class Fuse(nn.Module):
    def forward(self, features):
        return torch.cat(features, dim=-1)
