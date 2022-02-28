import numpy as np
import torch


class Transform:
    def __call__(self, obs):
        raise NotImplementedError

    def reset(self):
        pass


class Identity(Transform):
    def __call__(self, obs):
        return obs


class Compose(Transform):
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, obs):
        for transform in self._transforms:
            obs = transform(obs)
        return obs

    def reset(self):
        for transform in self._transforms:
            transform.reset()


class AsType(Transform):
    def __init__(self, dtype=np.float32):
        self._dtype = dtype

    def __call__(self, obs):
        return obs.astype(self._dtype)


class FrameStack(Transform):
    def __init__(self, frame_dim):
        """ stack observation along axis 0. Assumes observation has 1 less dimension
        """
        assert len(frame_dim) > 1
        self._frame_dim = frame_dim
        self._frames = np.zeros(shape=frame_dim, dtype=np.float32)

    def __call__(self, obs):
        self._frames = np.concatenate((self._frames[1:], [obs]))
        return self._frames

    def reset(self):
        self._frames.fill(0)
