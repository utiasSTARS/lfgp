import numpy as np
import torch

from torch.distributions import Normal, Uniform


class GripperIntentions:
    """
    Intentions that respectively open and close the gripper while following Gaussian noise for other actions
    """
    def __init__(self, action_dim, gripper_dim, means, vars):
        self._action_dim = action_dim
        self._gripper_dim = gripper_dim
        self._non_gripper_dim = torch.ones(self._action_dim)
        self._non_gripper_dim[gripper_dim] = 0
        self._non_gripper_dim = torch.where(self._non_gripper_dim)

        self.action_dist = Normal(loc=means, scale=vars)
        self.gripper_dist = Uniform(0., 1.)

        self._entropies = np.zeros((2, self._action_dim))
        self._means = np.zeros((2, self._action_dim))
        self._vars = np.ones((2, self._action_dim))
        for idx in range(2):
            self._entropies[idx][self._non_gripper_dim] = self.action_dist.entropy().numpy()
            self._entropies[idx, self._gripper_dim] = self.gripper_dist.entropy().numpy()
            self._means[idx][self._non_gripper_dim] = self.action_dist.mean.numpy()
            self._means[idx, self._gripper_dim]= self.gripper_dist.mean.numpy() * ((-1) ** (idx + 1))
            self._vars[idx][self._non_gripper_dim] = self.action_dist.variance.numpy()
            self._vars[idx, self._gripper_dim]= self.gripper_dist.variance.numpy()

    def compute_action(self, x, h):
        act = torch.zeros((2, self._action_dim), dtype=torch.float)
        log_probs = np.zeros((2, self._action_dim))

        for idx in range(2):
            act[idx][self._non_gripper_dim] = self.action_dist.sample()
            act[idx, self._gripper_dim] = 1 - self.gripper_dist.sample()
            log_probs[idx][self._non_gripper_dim] = self.action_dist.log_prob(act[idx][self._non_gripper_dim]).numpy()
            log_probs[idx, self._gripper_dim] = self.gripper_dist.log_prob(act[idx, self._gripper_dim]).numpy()

        act[0, self._gripper_dim] *= -1.
        log_probs = np.sum(log_probs, axis=-1)

        return act, np.zeros(2), h[0].cpu().numpy(), log_probs, self._entropies, self._means, self._vars

    def deterministic_action(self, x, h):
        act = torch.zeros((2, self._action_dim), dtype=torch.float)
        log_probs = np.zeros((2, self._action_dim))

        for idx in range(2):
            act[idx][self._non_gripper_dim] = self.action_dist.mean
            act[idx, self._gripper_dim] = 1.
            log_probs[idx][self._non_gripper_dim] = self.action_dist.log_prob(act[idx][self._non_gripper_dim]).numpy()
            log_probs[idx, self._gripper_dim] = self.gripper_dist.log_prob(act[idx, self._gripper_dim]).numpy()

        act[0, self._gripper_dim] *= -1.
        log_probs = np.sum(log_probs, axis=-1)

        return act, np.zeros(2), h[0].cpu().numpy(), log_probs, self._entropies, self._means, self._vars
