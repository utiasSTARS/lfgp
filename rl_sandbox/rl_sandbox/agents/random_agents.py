import numpy as np

import rl_sandbox.constants as c


class UniformContinuousAgent:
    def __init__(self, min_action, max_action, rng=np.random):
        self.min_action = np.array(min_action)
        self.max_action = np.array(max_action)
        self.entropy = np.log(max_action - min_action).astype(np.float32)
        self.log_prob = -self.entropy.sum(keepdims=True).astype(np.float32)
        self._act_info = {
            c.LOG_PROB: self.log_prob,
            c.ENTROPY: self.entropy,
            c.VALUE: np.array([np.nan], dtype=np.float32),
            c.MEAN: ((max_action + min_action) / 2).astype(np.float32),
            c.VARIANCE: (((max_action - min_action) ** 2) / 2).astype(np.float32),
        }
        self.rng = rng

    def compute_action(self, **kwargs):
        return self.rng.uniform(self.min_action, self.max_action).astype(np.float32), None, self._act_info

    def reset(self):
        return None


class UniformContinuousActionRepeatAgent(UniformContinuousAgent):
    def __init__(self, min_action, max_action, max_repeat, min_repeat=1, rng=np.random):
        super().__init__(min_action, max_action, rng)
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat
        self._cur_action = super().compute_action()
        self.reset()

    def compute_action(self, **kwargs):
        if self._ts >= self._action_repeat:
            self._cur_action = super().compute_action()
            self._action_repeat = self.rng.randint(self.min_repeat, self.max_repeat)
            self._ts = 0
        self._ts += 1

        return self._cur_action

    def reset(self):
        self._ts = 0
        self._action_repeat = self.rng.randint(self.min_repeat, self.max_repeat)
        return None