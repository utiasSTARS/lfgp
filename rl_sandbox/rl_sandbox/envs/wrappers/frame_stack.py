import numpy as np

from collections import deque

from rl_sandbox.envs.wrappers.wrapper import Wrapper


class FrameStackWrapper(Wrapper):
    def __init__(self, env, num_frames):
        assert num_frames > 0
        super().__init__(env)
        self._num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

    def _get_obs(self):
        assert len(self.frames) == self._num_frames
        # return np.stack(self.frames)
        return np.concatenate(self.frames)[None, :]

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        for _ in range(self._num_frames):
            self.frames.append(obs)

        return self._get_obs()

    def step(self, action, **kwargs):
        obs, reward, done, info = self._env.step(action, **kwargs)
        self.frames.append(obs)

        return self._get_obs(), reward, done, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def seed(self, seed):
        self._env.seed(seed)


class OfflineFrameStack:
    def __init__(self, num_frames):
        self._num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

    def get_stacked_obs(self, obs):
        assert len(self.frames) == self._num_frames
        self.frames.append(obs)
        # return np.stack(self.frames)
        return np.concatenate(self.frames)

    def reset(self, obs):
        for _ in range(self._num_frames):
            self.frames.append(obs)

        return np.concatenate(self.frames)


def make_frame_stack(num_frames, obss, dones, next_obss=None):
    # inefficiently doing this with a for loop for now
    stacked_obss = []
    frame_stacker = OfflineFrameStack(num_frames)
    if next_obss is not None:
        stacked_next_obss = []
        next_obss_frame_stacker = OfflineFrameStack(num_frames)

    new_ep = True

    for i in range(0, len(obss)):
        if new_ep:
            stacked_obss.append(frame_stacker.reset(obss[i]))
            if next_obss is not None:
                stacked_next_obss.append(next_obss_frame_stacker.reset(next_obss[i]))
        else:
            stacked_obss.append(frame_stacker.get_stacked_obs(obss[i]))
            if next_obss is not None:
                stacked_next_obss.append(next_obss_frame_stacker.get_stacked_obs(next_obss[i]))

        new_ep = dones[i]

    if next_obss is None:
        return np.vstack(stacked_obss)
    else:
        return np.vstack(stacked_obss), np.vstack(stacked_next_obss)