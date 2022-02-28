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


class NumPyBuffer(Buffer):
    """ The whole buffer is stored in RAM.
    """

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
                 rng=np.random,
                 dtype=np.float32):
        self.rng = rng
        self._memory_size = memory_size
        self._dtype = dtype
        self.observations = np.zeros(shape=(memory_size, *obs_dim), dtype=dtype)
        self.hidden_states = np.zeros(shape=(memory_size, *h_state_dim), dtype=np.float32)
        self.actions = np.zeros(shape=(memory_size, *action_dim), dtype=np.float32)
        self.rewards = np.zeros(shape=(memory_size, *reward_dim), dtype=np.float32)
        self.dones = np.zeros(shape=(memory_size, 1), dtype=np.bool)
        self.infos = dict()
        for info_name, (info_shape, info_dtype) in infos.items():
            self.infos[info_name] = np.zeros(shape=(memory_size, *info_shape), dtype=info_dtype)

        # This keeps track of the past X observations and hidden states for RNN
        self.burn_in_window = burn_in_window
        if burn_in_window > 0:
            self.padding_first = padding_first
            self.historic_observations = np.zeros(shape=(burn_in_window, *obs_dim), dtype=dtype)
            self.historic_hidden_states = np.zeros(shape=(burn_in_window, *h_state_dim), dtype=dtype)
            self.historic_dones = np.zeros(shape=(burn_in_window, 1), dtype=np.bool)

        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_idxes = np.ones(shape=memory_size, dtype=np.bool)
        if checkpoint_path is not None and memory_size >= checkpoint_interval > 0:
            self._checkpoint_path = checkpoint_path
            os.makedirs(checkpoint_path, exist_ok=True)
            self.checkpoint = self._checkpoint
            self._checkpoint_count = 0
        else:
            self.checkpoint = lambda: None

        self._pointer = 0
        self._count = 0

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

    def push(self, obs, h_state, act, rew, done, info, **kwargs):
        # Stores the overwritten observation and hidden state into historic variables
        if self.burn_in_window > 0:
            self.historic_observations = np.concatenate(
                (self.historic_observations[1:], [self.observations[self._pointer]]))
            self.historic_hidden_states = np.concatenate(
                (self.historic_hidden_states[1:], [self.hidden_states[self._pointer]]))
            self.historic_dones = np.concatenate(
                (self.historic_dones[1:], [self.dones[self._pointer]]))

        self.observations[self._pointer] = obs
        self.hidden_states[self._pointer] = h_state
        self.actions[self._pointer] = act
        self.rewards[self._pointer] = rew
        self.dones[self._pointer] = done
        self._checkpoint_idxes[self._pointer] = 0
        for info_name, info_value in info.items():
            if info_name not in self.infos:
                continue
            self.infos[info_name][self._pointer] = info_value

        self._pointer = (self._pointer + 1) % self._memory_size
        self._count += 1

        if self._checkpoint_interval > 0 and (self._memory_size - self._checkpoint_idxes.sum()) >= self._checkpoint_interval:
            self.checkpoint()
        return True

    def push_multiple(self, obss, h_states, acts, rews, dones, infos, **kwargs):
        if self.burn_in_window > 0:
            raise NotImplementedError("push_multiple not implemented for burn_in_window.")

        num_new = len(obss)
        if self._pointer + num_new > self.memory_size:
            indices = np.concatenate([range(self._pointer, self.memory_size),
                                      range(0, num_new - (self.memory_size - self._pointer))])
        else:
            indices = np.arange(self._pointer, stop=self._pointer + num_new)

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

    def _get_burn_in_window(self, idxes):
        historic_observations = np.zeros((len(idxes), self.burn_in_window, *self.observations.shape[1:]))
        historic_hidden_states = np.zeros((len(idxes), self.burn_in_window, *self.hidden_states.shape[1:]))
        not_dones = np.ones(len(idxes), dtype=np.bool)
        lengths = np.zeros(len(idxes), dtype=np.int)

        for ii in range(1, self.burn_in_window + 1):
            shifted_idxes = idxes - self._pointer - ii
            historic_idxes = np.logical_and(idxes - self._pointer - ii >= -self.burn_in_window, shifted_idxes < 0).astype(np.int)
            non_historic_idxes = 1 - historic_idxes

            not_dones[np.where(self.dones[idxes - ii, 0] * non_historic_idxes)] = 0
            not_dones[np.where(self.historic_dones[-ii, 0] * historic_idxes)] = 0
            non_historic_not_dones = not_dones * non_historic_idxes
            historic_not_dones = not_dones * historic_idxes

            historic_observations[:, -ii] += self.observations[idxes - ii] * non_historic_not_dones.reshape((-1, *([1] * len(historic_observations.shape[2:]))))
            historic_hidden_states[:, -ii] += self.hidden_states[idxes - ii] * non_historic_not_dones.reshape((-1, *([1] * len(historic_hidden_states.shape[2:]))))
            lengths += 1 * non_historic_not_dones

            if self._count > self._memory_size:
                historic_observations[:, -ii] += self.historic_observations[-self._pointer - ii] * historic_not_dones.reshape((-1, *([1] * len(historic_observations.shape[2:]))))
                historic_hidden_states[:, -ii] += self.historic_hidden_states[-self._pointer - ii] * historic_not_dones.reshape((-1, *([1] * len(historic_hidden_states.shape[2:]))))
                lengths += 1 * historic_not_dones

        return historic_observations, historic_hidden_states, lengths

    def get_transitions(self, idxes):
        obss = self.observations[idxes]
        h_states = self.hidden_states[idxes]
        acts = self.actions[idxes]
        rews = self.rewards[idxes]
        dones = self.dones[idxes]
        infos = {info_name: info_value[idxes] for info_name, info_value in self.infos.items()}
        
        lengths = np.ones(len(obss), dtype=np.int)
        if self.burn_in_window:
            historic_obss, historic_h_states, lengths = self._get_burn_in_window(idxes)
            obss = np.concatenate((historic_obss, obss[:, None, ...]), axis=1)
            h_states = np.concatenate((historic_h_states, h_states[:, None, ...]), axis=1)
            lengths += 1

            if self.padding_first:
                obss = np.array([np.roll(history, shift=(length % (self.burn_in_window + 1)), axis=0) for history, length in zip(obss, lengths)])
                h_states = np.array([np.roll(history, shift=(length % (self.burn_in_window + 1)), axis=0) for history, length in zip(h_states, lengths)])
        else:
            obss = obss[:, None, ...]
            h_states = h_states[:, None, ...]

        return obss, h_states, acts, rews, dones, infos, lengths

    def get_next(self, next_idxes, next_obs, next_h_state):
        # Replace all indices that are equal to memory size to zero
        idxes_to_modify = np.where(next_idxes == len(self))[0]
        next_idxes[idxes_to_modify] = 0
        next_obss = self.observations[next_idxes]
        next_h_states = self.hidden_states[next_idxes]

        # Replace the content for the sample at current timestep
        idxes_to_modify = np.where(next_idxes == self._pointer)[0]
        next_obss[idxes_to_modify] = next_obs
        next_h_states[idxes_to_modify] = next_h_state

        return next_obss, next_h_states

    def sample(self, batch_size, idxes=None):
        if not len(self):
            raise NoSampleError

        if idxes is None:
            random_idxes = self.rng.randint(len(self), size=batch_size)
        else:
            random_idxes = idxes

        obss, h_states, acts, rews, dones, infos, lengths = self.get_transitions(random_idxes)

        return obss, h_states, acts, rews, dones, infos, lengths, random_idxes

    def sample_with_next_obs(self, batch_size, next_obs, next_h_state, idxes=None):
        obss, h_states, acts, rews, dones, infos, lengths, random_idxes = NumPyBuffer.sample(self, batch_size, idxes)

        next_idxes = random_idxes + 1
        next_obss, next_h_states = self.get_next(next_idxes, next_obs, next_h_state)
        next_obss = next_obss[:, None, ...]
        next_h_states = next_h_states[:, None, ...]

        return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, random_idxes

    def sample_init_obs(self, batch_size):
        if np.sum(self.dones) == 0:
            raise NoSampleError
        init_idxes = (np.where(self.dones == 1)[0] + 1) % self._memory_size

        # Filter out indices greater than the current pointer.
        # It is useless once we exceed count > memory size
        init_idxes = init_idxes[init_idxes < len(self)]

        # Filter out the sample at is being pointed because we can't tell confidently that it is initial state
        init_idxes = init_idxes[init_idxes != self._pointer]
        random_idxes = init_idxes[self.rng.randint(len(init_idxes), size=batch_size)]
        return self.observations[random_idxes], self.hidden_states[random_idxes]

    def sample_consecutive(self, batch_size, end_with_done=False):
        batch_size = min(batch_size, len(self))

        if end_with_done:
            valid_idxes = np.where(self.dones == 1)[0]
            valid_idxes = valid_idxes[valid_idxes + 1 >= batch_size]
            if len(valid_idxes) <= 0:
                raise NoSampleError
            random_ending_idx = self.rng.choice(valid_idxes) + 1
        else:
            random_ending_idx = self.rng.randint(batch_size, len(self) + 1)

        idxes = np.arange(random_ending_idx - batch_size, random_ending_idx)
        obss, h_states, acts, rews, dones, infos, lengths = self.get_transitions(idxes)
        
        return obss, h_states, acts, rews, dones, infos, lengths, random_ending_idx

    def save(self, save_path, end_with_done=True):
        pointer = self._pointer
        count = self._count

        if end_with_done:
            done_idxes = np.where(self.dones == 1)[0]
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

        with gzip.open(save_path, "wb") as f:
            pickle.dump({
                c.OBSERVATIONS: self.observations,
                c.HIDDEN_STATES: self.hidden_states,
                c.ACTIONS: self.actions,
                c.REWARDS: self.rewards,
                c.DONES: self.dones,
                c.INFOS: self.infos,
                c.MEMORY_SIZE: self._memory_size,
                c.POINTER: pointer,
                c.COUNT: count,
                c.DTYPE: self._dtype,
                c.RNG: self.rng,
            }, f)

    def load(self, load_path):
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)

        self._memory_size = data[c.MEMORY_SIZE]
        self.observations = data[c.OBSERVATIONS]
        self.hidden_states = data[c.HIDDEN_STATES]
        self.actions = data[c.ACTIONS]
        self.rewards = data[c.REWARDS]
        self.dones = data[c.DONES]
        self.infos = data[c.INFOS]

        self._pointer = data[c.POINTER]
        self._count = data[c.COUNT]

        self._dtype = data[c.DTYPE]
        self.rng = data[c.RNG]

    def transfer_data(self, load_path):
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)

        assert data[c.COUNT] <= data[c.MEMORY_SIZE] or data[c.COUNT] == data[c.POINTER]

        count = data[c.COUNT]
        self.observations[:count] = data[c.OBSERVATIONS][-count:]
        self.hidden_states[:count] = data[c.HIDDEN_STATES][-count:]
        self.actions[:count] = data[c.ACTIONS][-count:]
        self.rewards[:count] = data[c.REWARDS][-count:]
        self.dones[:count] = data[c.DONES][-count:]
        
        self._pointer = data[c.POINTER]
        self._count = data[c.COUNT]

        for k, v in self.infos.items():
            self.infos[k][:count] = data[c.INFOS][k][-count:]

    def get_next_state_buffer(self, downsample_rate=1.0, max_index=None):
        if max_index is not None:
            assert max_index <= len(self)
        else:
            max_index = len(self)

        if downsample_rate < 1.0:
            inds = self.rng.choice(range(max_index), int(downsample_rate * max_index), replace=False)
        else:
            inds = np.array(range(max_index))
        new_buf_size = len(inds)

        obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, _, _ = \
            self.sample_with_next_obs(None, None, None, idxes=inds)

        if hasattr(self, '_checkpoint_path'):
            cpath = self._checkpoint_path
        else:
            cpath = None

        infos_init = dict.fromkeys(self.infos.keys())
        for k in infos_init:
            infos_init[k] = (self.infos[k].shape[1:], self.infos[k].dtype)

        new_buffer = NextStateNumPyBuffer(self._memory_size, self.observations.shape[1:],
                                          self.hidden_states.shape[1:], self.actions.shape[1:], self.rewards.shape[1:],
                                          infos_init, self._checkpoint_interval, cpath,
                                          self.rng, self._dtype)

        new_buffer.observations[:new_buf_size] = obss.squeeze(axis=1)
        new_buffer.hidden_states[:new_buf_size] = h_states.squeeze(axis=1)
        new_buffer.actions[:new_buf_size] = acts
        new_buffer.rewards[:new_buf_size] = rews
        new_buffer.dones[:new_buf_size] = dones
        new_buffer.next_observations[:new_buf_size] = next_obss.squeeze(axis=1)
        new_buffer.next_hidden_states[:new_buf_size] = next_h_states.squeeze(axis=1)
        for k in self.infos.keys():
            new_buffer.infos[k][:new_buf_size] = infos[k]
        new_buffer._pointer = new_buf_size
        new_buffer._count = new_buf_size

        return new_buffer


class NextStateNumPyBuffer(NumPyBuffer):
    """ The whole buffer is stored in RAM.
    """

    def __init__(self,
                 memory_size,
                 obs_dim,
                 h_state_dim,
                 action_dim,
                 reward_dim,
                 infos=dict(),
                 checkpoint_interval=0,
                 checkpoint_path=None,
                 rng=np.random,
                 dtype=np.float32):
        super().__init__(memory_size=memory_size,
                         obs_dim=obs_dim,
                         h_state_dim=h_state_dim,
                         action_dim=action_dim,
                         reward_dim=reward_dim,
                         infos=infos,
                         checkpoint_interval=checkpoint_interval,
                         checkpoint_path=checkpoint_path,
                         rng=rng,
                         dtype=dtype)
        self.next_observations = self.observations.copy()
        self.next_hidden_states = self.hidden_states.copy()

    def __getitem__(self, index):
        return (self.observations[index],
                self.hidden_states[index],
                self.actions[index],
                self.rewards[index],
                self.dones[index],
                self.next_observations[index],
                {info_name: info_value[index] for info_name, info_value in self.infos.items()})

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
                c.NEXT_OBSERVATIONS: self.next_observations[transitions_to_save],
                c.NEXT_HIDDEN_STATES: self.next_hidden_states[transitions_to_save],
                c.INFOS: {info_name: info_value[transitions_to_save] for info_name, info_value in self.infos.items()},
            }, f)

        self._checkpoint_idxes[transitions_to_save] = 1
        self._checkpoint_count += 1

    def push(self, obs, h_state, act, rew, done, info, next_obs, next_h_state, **kwargs):
        self.next_observations[self._pointer] = next_obs
        self.next_hidden_states[self._pointer] = next_h_state

        return super().push(obs=obs, h_state=h_state, act=act, rew=rew, done=done, info=info)

    def push_multiple(self, obss, h_states, acts, rews, dones, infos, next_obss, next_h_states, **kwargs):
        num_new = len(obss)
        if self._pointer + num_new > self.memory_size:
            indices = np.concatenate([range(self._pointer, self.memory_size),
                                      range(0, num_new - (self.memory_size - self._pointer))])
        else:
            indices = np.arange(self._pointer, stop=self._pointer + num_new)

        self.next_observations[indices] = next_obss.squeeze(1)
        self.next_hidden_states[indices] = next_h_states.squeeze(1)

        return super().push_multiple(obss=obss, h_states=h_states, acts=acts, rews=rews, dones=dones, infos=infos)

    def sample_with_next_obs(self, batch_size, next_obs, next_h_state, idxes=None):
        obss, h_states, acts, rews, dones, infos, lengths, random_idxes = NumPyBuffer.sample(self, batch_size, idxes)
        next_obss = self.next_observations[random_idxes][:, None, ...]
        next_h_states = self.next_hidden_states[random_idxes][:, None, ...]

        return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, random_idxes

    def save(self, save_path, end_with_done=True):
        pointer = self._pointer
        count = self._count

        if end_with_done:
            done_idxes = np.where(self.dones == 1)[0]
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

        with gzip.open(save_path, "wb") as f:
            pickle.dump({
                c.OBSERVATIONS: self.observations,
                c.HIDDEN_STATES: self.hidden_states,
                c.ACTIONS: self.actions,
                c.REWARDS: self.rewards,
                c.DONES: self.dones,
                c.NEXT_OBSERVATIONS: self.next_observations,
                c.NEXT_HIDDEN_STATES: self.next_hidden_states,
                c.INFOS: self.infos,
                c.MEMORY_SIZE: self._memory_size,
                c.POINTER: pointer,
                c.COUNT: count,
                c.DTYPE: self._dtype,
                c.RNG: self.rng,
            }, f)

    def load(self, load_path):
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

        self._dtype = data[c.DTYPE]
        if c.RNG in data.keys():
            self.rng = data[c.RNG]
        else:
            self.rng = np.random

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
            inds = self.rng.choice(range(max_index), int(downsample_rate * max_index), replace=False)
        else:
            inds = range(max_index)
        new_buf_size = len(inds)

        items = ['observations', 'hidden_states', 'actions', 'rewards', 'dones', 'next_observations',
                 'next_hidden_states']

        for it in items:
            old_data = getattr(self, it)
            setattr(self, it, np.zeros_like(old_data))
            getattr(self, it)[:new_buf_size] = old_data[inds]

        for k in self.infos.keys():
            old_data = self.infos[k]
            self.infos[k] = np.zeros_like(old_data)
            self.infos[k][:new_buf_size] = old_data[inds]

        self._pointer = new_buf_size
        self._count = new_buf_size


class TrajectoryNumPyBuffer(NumPyBuffer):
    """ This buffer stores one trajectory as a sample
    """
