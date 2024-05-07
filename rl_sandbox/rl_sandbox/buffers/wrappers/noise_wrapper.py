import torch

import rl_sandbox.constants as c

from rl_sandbox.buffers.wrappers.buffer_wrapper import BufferWrapper
from rl_sandbox.model_architectures.utils import RunningMeanStd


class NoiseBuffer(BufferWrapper):
    def __init__(self, buffer, noise_magnitude, update_on_sample=False):
        super().__init__(buffer)

        self._noise_mag = noise_magnitude
        self._obs_mean = None
        self._obs_std = None
        self._dim_noise_mag = None
        if update_on_sample:
            self._rms = None

    def sample(self, *args, **kwargs):
        buf_data = super().sample(*args, **kwargs)
        obss = buf_data[0]

        if self._dim_noise_mag is None and not hasattr(self, '_rms'): self.update_stats()

        if hasattr(self, '_rms'):
            if self._rms is None:
                self._rms = RunningMeanStd(shape=(obss.shape[-1],), device=self.device)
            self._rms.update(obss)
            self._obs_mean = self._rms.mean
            self._obs_std = self._rms.std
            self._dim_noise_mag = self._noise_mag * self._obs_std


        obss_noise = torch.randn_like(obss) * self._dim_noise_mag
        obss += obss_noise  # changes buf_data as well

        return buf_data

    def sample_with_next_obs(self, *args, **kwargs):
        buf_data = super().sample_with_next_obs(*args, **kwargs)
        obss = buf_data[0]
        next_obss = buf_data[5]

        if self._dim_noise_mag is None and not hasattr(self, '_rms'): self.update_stats()

        if hasattr(self, '_rms'):
            if self._rms is None:
                self._rms = RunningMeanStd(shape=(obss.shape[-1],), device=self.device)
            self._rms.update(obss)
            self._obs_mean = self._rms.mean
            self._obs_std = self._rms.std
            self._dim_noise_mag = self._noise_mag * self._obs_std

        obss_noise = torch.randn_like(obss) * self._dim_noise_mag
        next_obss_noise = torch.randn_like(next_obss) * self._dim_noise_mag
        obss += obss_noise  # changes buf_data as well
        next_obss += next_obss_noise  # changes buf_data as well

        return buf_data

    def sample_trajs(self, *args, **kwargs):
        buf_data = super().sample_trajs(*args, **kwargs)
        obss = buf_data[0]
        next_obss = buf_data[5]  # TODO this only works with the way torch pin buffer is set up for now

        if self._dim_noise_mag is None and not hasattr(self, '_rms'): self.update_stats()

        if hasattr(self, '_rms'):
            if self._rms is None:
                self._rms = RunningMeanStd(shape=(obss.shape[-1],), device=self.device)
            self._rms.update(obss)
            self._obs_mean = self._rms.mean
            self._obs_std = self._rms.std
            self._dim_noise_mag = self._noise_mag * self._obs_std


        obss_noise = torch.randn_like(obss) * self._dim_noise_mag
        next_obss_noise = torch.randn_like(next_obss) * self._dim_noise_mag
        obss += obss_noise  # changes buf_data as well
        next_obss += next_obss_noise  # changes buf_data as well

        return buf_data

    def update_stats(self):
        self._obs_mean = self.buffer.observations.mean(axis=0)
        self._obs_std = self.buffer.observations.std(axis=0)
        self._dim_noise_mag = self._noise_mag * self._obs_std
