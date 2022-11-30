import _pickle as pickle
import gzip
import numpy as np
import os
import torch

from collections import deque

from rl_sandbox.buffers.buffer import (
    Buffer,
    CheckpointIndexError,
    LengthMismatchError,
    NoSampleError
)
import rl_sandbox.constants as c


class TorchPinBuffer(Buffer):
    def __init__(self,
                 memory_size,
                 obs_dim,
                 h_state_dim,
                 action_dim,
                 reward_dim,
                 infos=dict(),
                 burn_in_window=0,
                 padding_first=False,
                 checkpoint_interval=0,
                 checkpoint_path=None,
                 dtype=torch.float32, 
                 device='cpu',
                 rng=None):
        self._memory_size = memory_size
        self._dtype = dtype
        self.observations = torch.zeros(size=(memory_size, *obs_dim), dtype=dtype, device=device)
        self.next_observations = torch.zeros(size=(memory_size, *obs_dim), dtype=dtype, device=device)
        self.hidden_states = torch.zeros(size=(memory_size, *h_state_dim), dtype=dtype, device=device)
        self.next_hidden_states = torch.zeros(size=(memory_size, *h_state_dim), dtype=dtype, device=device)
        self.actions = torch.zeros(size=(memory_size, *action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size=(memory_size, *reward_dim), dtype=torch.float32)
        self.dones = torch.zeros(size=(memory_size, 1), dtype=torch.int, device=device)
        self.infos = dict()
        for info_name, (info_shape, info_dtype) in infos.items():
            self.infos[info_name] = np.zeros(shape=(memory_size, *info_shape), dtype=info_dtype)
            self.infos[info_name] = torch.as_tensor(self.infos[info_name], device=device)

        # This keeps track of the past X observations and hidden states for RNN
        self.burn_in_window = burn_in_window
        if burn_in_window > 0:
            self.padding_first = padding_first
            self.historic_observations = torch.zeros(size=(burn_in_window, *obs_dim), dtype=dtype, device=device)
            self.historic_hidden_states = torch.zeros(size=(burn_in_window, *h_state_dim), dtype=dtype, device=device)
            self.historic_dones = torch.zeros(size=(burn_in_window, 1), dtype=torch.bool)

        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_idxes = torch.ones(size=(memory_size,), dtype=torch.bool, device=device)
        if checkpoint_path is not None and memory_size >= checkpoint_interval > 0:
            self._checkpoint_path = checkpoint_path
            os.makedirs(checkpoint_path, exist_ok=True)
            self.checkpoint = self._checkpoint
            self._checkpoint_count = 0
        else:
            self.checkpoint = lambda: None

        self._pointer = 0
        self._count = 0
        self.device = device
    
    @property
    def memory_size(self):
        return self._memory_size

    @property
    def is_full(self):
        return self._count >= self.memory_size

    @property
    def pointer(self):
        return self._pointer
    
    def __getitem__(self, index):
        return (self.observations[index],
                self.hidden_states[index],
                self.actions[index],
                self.rewards[index],
                self.dones[index],
                {info_name: info_value[index] for info_name, info_value in self.infos.items()})
    
    def __len__(self):
        return min(self._count, self.memory_size)
    
    def _convert_all_to_torch(self, include_next_obs=True):
        for data_str in ['observations', 'hidden_states', 'actions', 'rewards']:
            setattr(self, data_str, torch.as_tensor(getattr(self, data_str), device=self.device).float())
        self.dones = torch.as_tensor(self.dones, device=self.device).long()
        self.infos = {k: torch.as_tensor(v, device=self.device) for k, v in self.infos.items()}
        if include_next_obs:
            for data_str in ['next_observations', 'next_hidden_states']:
                setattr(self, data_str, torch.as_tensor(getattr(self, data_str), device=self.device).float())

    def _checkpoint(self):
        transitions_to_save = np.where(self._checkpoint_idxes == 0)[0]

        if len(transitions_to_save) == 0:
            return

        idx_diff = np.where(transitions_to_save - np.concatenate(([transitions_to_save[0] - 1], transitions_to_save[:-1])) > 1)[0]

        if len(idx_diff) > 1:
            raise CheckpointIndexError
        elif len(idx_diff) == 1:
            transitions_to_save = np.concatenate((transitions_to_save[idx_diff[0]:], transitions_to_save[:idx_diff[0]]))

        with gzip.open(os.path.join(f"{self._checkpoint_path}", f"{self._checkpoint_count}.pkl"), "wb") as f:
            pickle.dump({
                c.OBSERVATIONS: self.observations[transitions_to_save],
                c.HIDDEN_STATES: self.hidden_states[transitions_to_save],
                c.ACTIONS: self.actions[transitions_to_save],
                c.REWARDS: self.rewards[transitions_to_save],
                c.DONES: self.dones[transitions_to_save],
                c.INFOS: {info_name: info_value[transitions_to_save] for info_name, info_value in self.infos.items()},
            }, f)

        self._checkpoint_idxes[transitions_to_save] = 1
        self._checkpoint_count += 1
    
    def push(self, obs, h_state, act, rew, done, info, next_obs, next_h_state, **kwargs):
        # Stores the overwritten observation and hidden state into historic variables
        if self.burn_in_window > 0:
            self.historic_observations = torch.cat(
                (self.historic_observations[1:], [self.observations[self._pointer]]))
            self.historic_hidden_states = torch.cat(
                (self.historic_hidden_states[1:], [self.hidden_states[self._pointer]]))
            self.historic_dones = torch.cat(
                (self.historic_dones[1:], [self.dones[self._pointer]]))

        self.next_observations[self._pointer] = torch.as_tensor(next_obs, dtype=self._dtype, device=self.device)
        self.next_hidden_states[self._pointer] = torch.as_tensor(next_h_state, dtype=self._dtype, device=self.device)
        self.observations[self._pointer] = torch.as_tensor(obs, dtype=self._dtype, device=self.device)
        self.hidden_states[self._pointer] = torch.as_tensor(h_state, dtype=self._dtype, device=self.device)
        self.actions[self._pointer] = torch.as_tensor(act, dtype=self._dtype, device=self.device)
        self.rewards[self._pointer] = torch.as_tensor(rew, dtype=self._dtype, device=self.device)
        self.dones[self._pointer] = torch.as_tensor(done, dtype=self._dtype, device=self.device)
        self._checkpoint_idxes[self._pointer] = 0
        for info_name, info_value in info.items():
            if info_name not in self.infos:
                continue
            self.infos[info_name][self._pointer] = torch.as_tensor(info_value, dtype=self._dtype, device=self.device)

        self._pointer = (self._pointer + 1) % self._memory_size
        self._count += 1

        if self._checkpoint_interval > 0 and (self._memory_size - self._checkpoint_idxes.sum()) >= self._checkpoint_interval:
            self.checkpoint()
        return True
    
    def push_multiple(self, obss, h_states, acts, rews, dones, infos, next_obss, next_h_states, **kwargs):
        num_new = len(obss)
        if self._pointer + num_new > self.memory_size:
            indices = torch.cat([range(self._pointer, self.memory_size),
                                      range(0, num_new - (self.memory_size - self._pointer))])
        else:
            indices = torch.arange(self._pointer, stop=self._pointer + num_new)

        self.next_observations[indices] = next_obss.squeeze(1)
        self.next_hidden_states[indices] = next_h_states.squeeze(1)

        if self.burn_in_window > 0:
            raise NotImplementedError("push_multiple not implemented for burn_in_window.")

        num_new = len(obss)
        if self._pointer + num_new > self.memory_size:
            indices = torch.cat([range(self._pointer, self.memory_size),
                                      range(0, num_new - (self.memory_size - self._pointer))])
        else:
            indices = torch.arange(self._pointer, stop=self._pointer + num_new)

        self.observations[indices] = obss.squeeze(1)
        self.hidden_states[indices] = h_states.squeeze(1)
        self.actions[indices] = acts
        self.rewards[indices] = rews
        self.dones[indices] = dones
        self._checkpoint_idxes[indices] = 0
        for info_name, info_value in infos.items():
            if info_name not in self.infos:
                continue
            self.infos[info_name][indices] = info_value

        self._pointer = (self._pointer + num_new) % self._memory_size
        self._count += num_new

        if self._checkpoint_interval > 0 and (self._memory_size - self._checkpoint_idxes.sum()) >= self._checkpoint_interval:
            self.checkpoint()
        return True
    
    def clear(self):
        self._pointer = 0
        self._count = 0
        self._checkpoint_idxes.fill(1)
    
    def get_transitions(self, idxes):
        obss = self.observations[idxes]
        h_states = self.hidden_states[idxes]
        acts = self.actions[idxes]
        rews = self.rewards[idxes]
        dones = self.dones[idxes]
        infos = {info_name: info_value[idxes] for info_name, info_value in self.infos.items()}
        
        lengths = torch.ones(len(obss), dtype=torch.long, device=self.device)
        if self.burn_in_window:
            raise NotImplementedError("Burn in Window Not implemented for TorchPinBuffer")
        else:
            obss = obss[:, None, ...]
            h_states = h_states[:, None, ...]

        return obss, h_states, acts, rews, dones, infos, lengths
        
    def _convert_all_to_torch(self, include_next_obs=True):
        for data_str in ['observations', 'hidden_states', 'actions', 'rewards']:
            setattr(self, data_str, torch.as_tensor(getattr(self, data_str), device=self.device).float())
        self.dones = torch.as_tensor(self.dones, device=self.device).long()
        self.infos = {k: torch.as_tensor(v, device=self.device) for k, v in self.infos.items()}
        if include_next_obs:
            for data_str in ['next_observations', 'next_hidden_states']:
                setattr(self, data_str, torch.as_tensor(getattr(self, data_str), device=self.device).float())
    
    def get_next(self, next_idxes, next_obs, next_h_state):
        # Replace all indices that are equal to memory size to zero
        idxes_to_modify = torch.where(next_idxes == len(self))[0]
        next_idxes[idxes_to_modify] = 0
        next_obss = self.observations[next_idxes]
        next_h_states = self.hidden_states[next_idxes]

        # Replace the content for the sample at current timestep
        idxes_to_modify = torch.where(next_idxes == self._pointer)[0]
        next_obss[idxes_to_modify] = next_obs
        next_h_states[idxes_to_modify] = next_h_state

        return next_obss, next_h_states
    
    def sample(self, batch_size, idxes=None, include_random_idxes=False):
        if not len(self):
            raise NoSampleError

        if idxes is None:
            random_idxes = torch.randint(len(self), size=(batch_size,))
        else:
            random_idxes = idxes

        obss, h_states, acts, rews, dones, infos, lengths = self.get_transitions(random_idxes)

        if include_random_idxes:
            return obss, h_states, acts, rews, dones, infos, lengths, random_idxes
        else:
            return obss, h_states, acts, rews, dones, infos, lengths

    def sample_with_next_obs(self, batch_size, next_obs, next_h_state=None, idxes=None):
        # all TorchPinBuffers are NextStateBuffers, so match NextStateNumPyBuffer
        obss, h_states, acts, rews, dones, infos, lengths, random_idxes = self.sample(batch_size, idxes, True)
        next_obss = self.next_observations[random_idxes][:, None, ...]
        next_h_states = self.next_hidden_states[random_idxes][:, None, ...]

        # return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, random_idxes
        return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths
    
    def sample_init_obs(self, batch_size):
        if torch.sum(self.dones) == 0:
            raise NoSampleError
        init_idxes = (torch.where(self.dones == 1)[0] + 1) % self._memory_size

        # Filter out indices greater than the current pointer.
        # It is useless once we exceed count > memory size
        init_idxes = init_idxes[init_idxes < len(self)]

        # Filter out the sample at is being pointed because we can't tell confidently that it is initial state
        init_idxes = init_idxes[init_idxes != self._pointer]
        random_idxes = init_idxes[torch.randint(len(init_idxes), size=(batch_size,))]
        return self.observations[random_idxes], self.hidden_states[random_idxes]

    def sample_consecutive(self, batch_size, end_with_done=False):
        batch_size = min(batch_size, len(self))

        if end_with_done:
            valid_idxes = torch.where(self.dones == 1)[0]
            valid_idxes = valid_idxes[valid_idxes + 1 >= batch_size]
            if len(valid_idxes) <= 0:
                raise NoSampleError
            random_ending_idx = torch.multinomial(valid_idxes, 1) + 1
        else:
            random_ending_idx = torch.randint(batch_size, (len(self) + 1,))

        idxes = torch.arange(random_ending_idx - batch_size, random_ending_idx)
        obss, h_states, acts, rews, dones, infos, lengths = self.get_transitions(idxes)
        
        return obss, h_states, acts, rews, dones, infos, lengths, random_ending_idx
    
    def save(self, save_path, end_with_done=True):
        # convert to numpy before saving, for consistencyi
        pointer = self._pointer
        count = self._count

        if end_with_done:
            done_idxes = torch.where(self.dones == 1)[0]
            if len(done_idxes) == 0:
                print("No completed episodes. Nothing to save.")
                return

            wraparound_idxes = done_idxes[done_idxes < self._pointer]
            if len(wraparound_idxes) > 0:
                pointer = (wraparound_idxes[-1] + 1) % self._memory_size
                count -= (self._pointer - pointer)
            else:
                pointer = (done_idxes[-1] + 1) % self._memory_size
                count -= (self._pointer + self._memory_size - pointer)
        
        cpu_infos = {k: v.cpu().numpy() for k, v in self.infos.items()}

        with gzip.open(save_path, "wb") as f:
            pickle.dump({
                c.OBSERVATIONS: self.observations.cpu().numpy(),
                c.HIDDEN_STATES: self.hidden_states.cpu().numpy(),
                c.ACTIONS: self.actions.cpu().numpy(),
                c.REWARDS: self.rewards.cpu().numpy(),
                c.DONES: self.dones.cpu().numpy(),
                c.NEXT_OBSERVATIONS: self.next_observations.cpu().numpy(),
                c.NEXT_HIDDEN_STATES: self.next_hidden_states.cpu().numpy(),
                c.INFOS: cpu_infos,
                c.MEMORY_SIZE: self._memory_size,
                c.POINTER: pointer,
                c.COUNT: count,
                c.DTYPE: np.float32
            }, f)

    def load(self, load_path, load_rng=True):
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)

        self._memory_size = data[c.MEMORY_SIZE]
        self.observations = data[c.OBSERVATIONS]
        self.hidden_states = data[c.HIDDEN_STATES]
        self.next_observations = data[c.NEXT_OBSERVATIONS]
        self.next_hidden_states = data[c.NEXT_HIDDEN_STATES]
        self.actions = data[c.ACTIONS]
        self.rewards = data[c.REWARDS]
        self.dones = data[c.DONES]
        self.infos = data[c.INFOS]

        self._pointer = data[c.POINTER]
        self._count = data[c.COUNT]

        self._dtype = torch.float32
        self._convert_all_to_torch()
    
    def transfer_data(self, load_path):
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)

        assert data[c.COUNT] <= data[c.MEMORY_SIZE] or data[c.COUNT] == data[c.POINTER]

        count = data[c.COUNT]
        self.observations[:count] = data[c.OBSERVATIONS][-count:]
        self.hidden_states[:count] = data[c.HIDDEN_STATES][-count:]
        self.next_observations[:count] = data[c.NEXT_OBSERVATIONS][-count:]
        self.next_hidden_states[:count] = data[c.NEXT_HIDDEN_STATES][-count:]
        self.actions[:count] = data[c.ACTIONS][-count:]
        self.rewards[:count] = data[c.REWARDS][-count:]
        self.dones[:count] = data[c.DONES][-count:]
        
        self._pointer = data[c.POINTER]
        self._count = data[c.COUNT]

        for k, v in self.infos.items():
            self.infos[k][:count] = data[c.INFOS][k][-count:]
    
    def downsample(self, downsample_rate, max_index=None):
        if max_index is not None:
            assert max_index <= len(self)
        else:
            max_index = len(self)
        if downsample_rate < 1:
            inds = torch.multinomial(range(max_index), int(downsample_rate * max_index), replacement=False)
        else:
            inds = range(max_index)
        new_buf_size = len(inds)

        items = ['observations', 'hidden_states', 'actions', 'rewards', 'dones', 'next_observations',
                 'next_hidden_states']

        for it in items:
            old_data = getattr(self, it)
            setattr(self, it, torch.zeros_like(old_data))
            getattr(self, it)[:new_buf_size] = old_data[inds]

        for k in self.infos.keys():
            old_data = self.infos[k]
            self.infos[k] = torch.zeros_like(old_data)
            self.infos[k][:new_buf_size] = old_data[inds]

        self._pointer = new_buf_size
        self._count = new_buf_size
