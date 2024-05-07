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
from rl_sandbox.envs.wrappers.frame_stack import make_frame_stack
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
        self.dones = np.zeros(shape=(memory_size, 1), dtype=bool)
        self.infos = dict()
        for info_name, (info_shape, info_dtype) in infos.items():
            self.infos[info_name] = np.zeros(shape=(memory_size, *info_shape), dtype=info_dtype)
        # This keeps track of the past X observations and hidden states for RNN
        self.burn_in_window = burn_in_window
        if burn_in_window > 0:
            self.padding_first = padding_first
            self.historic_observations = np.zeros(shape=(burn_in_window, *obs_dim), dtype=dtype)
            self.historic_hidden_states = np.zeros(shape=(burn_in_window, *h_state_dim), dtype=dtype)
            self.historic_dones = np.zeros(shape=(burn_in_window, 1), dtype=bool)

        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_idxes = np.ones(shape=memory_size, dtype=bool)
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
        not_dones = np.ones(len(idxes), dtype=bool)
        lengths = np.zeros(len(idxes), dtype=int)

        for ii in range(1, self.burn_in_window + 1):
            shifted_idxes = idxes - self._pointer - ii
            historic_idxes = np.logical_and(idxes - self._pointer - ii >= -self.burn_in_window, shifted_idxes < 0).astype(int)
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

        lengths = np.ones(len(obss), dtype=int)
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
                # c.RNG: self.rng,  # removing for now because it breaks across numpy versions, and we don't use it
            }, f)

    def load(self, load_path, load_rng=True, start_idx=0, end_idx=None, frame_stack=1):
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)

        if end_idx is None:
            self._memory_size = data[c.MEMORY_SIZE]
        else:
            self._memory_size = end_idx - start_idx

        if data['observations'].shape[-1] == self.observations.shape[-1] + 1:
            print("----------------------")
            print(f"Warning: data at {load_path} has obs dim 1 higher than configured. Assuming that the last"\
                  f"dim was an absorbing state index, and cutting it off during loading.")
            print("----------------------")
            data['observations'] = data['observations'][:, :-1]

        elif data['observations'].shape[-1] == self.observations.shape[-1] + 1 + 6:
            print("----------------------")
            print(f"Warning: data at {load_path} has obs dim 7 higher than configured. Assuming that the last "\
                  f"dim was an absorbing state index, and the last 6 before that were force torque, "\
                  f"and cutting all 7 off during loading.")
            print("----------------------")
            data['observations'] = data['observations'][:, :-7]

        if frame_stack > 1:
            self.observations = make_frame_stack(frame_stack,
                data[c.OBSERVATIONS][start_idx:end_idx], data[c.DONES][start_idx:end_idx])
        else:
            self.observations = data[c.OBSERVATIONS][start_idx:end_idx]

        self._memory_size = data[c.MEMORY_SIZE][start_idx:end_idx]
        self.hidden_states = data[c.HIDDEN_STATES][start_idx:end_idx]
        self.actions = data[c.ACTIONS][start_idx:end_idx]
        self.rewards = data[c.REWARDS][start_idx:end_idx]
        self.dones = data[c.DONES][start_idx:end_idx]

        for k in data[c.INFOS]:
            self.infos[k] = data[c.INFOS][k][start_idx:end_idx]

        self._pointer = data[c.POINTER]
        self._count = data[c.COUNT]

        # adjust if sizes change based on different idxes
        if end_idx is not None:
            self._count = min(self._count, end_idx - start_idx)
            self._pointer = self._pointer % len(self)

        self._dtype = data[c.DTYPE]
        if load_rng:
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
                # c.RNG: self.rng,  # removing for now because it breaks across numpy versions, and we don't use it
            }, f)

    def load(self, load_path, load_rng=True, start_idx=0, end_idx=None, frame_stack=1):
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)

        if end_idx is None:
            self._memory_size = data[c.MEMORY_SIZE]
        else:
            self._memory_size = end_idx - start_idx

        if data['observations'].shape[-1] == self.observations.shape[-1] + 1:
            print("----------------------")
            print(f"Warning: data at {load_path} has obs dim 1 higher than configured. Assuming that the last"\
                  f"dim was an absorbing state index, and cutting it off during loading.")
            print("----------------------")
            data['observations'] = data['observations'][:, :-1]
            data['next_observations'] = data['next_observations'][:, :-1]
        elif data['observations'].shape[-1] == self.observations.shape[-1] + 1 + 6:
            print("----------------------")
            print(f"Warning: data at {load_path} has obs dim 7 higher than configured. Assuming that the last "\
                  f"dim was an absorbing state index, and the last 6 before that were force torque, "\
                  f"and cutting all 7 off during loading.")
            print("----------------------")
            data['observations'] = data['observations'][:, :-7]
            data['next_observations'] = data['next_observations'][:, :-7]

        if frame_stack > 1:
            self.observations, self.next_observations = make_frame_stack(frame_stack,
                data[c.OBSERVATIONS][start_idx:end_idx], data[c.DONES][start_idx:end_idx],
                data[c.NEXT_OBSERVATIONS][start_idx:end_idx])
        else:
            self.observations = data[c.OBSERVATIONS][start_idx:end_idx]
            self.next_observations = data[c.NEXT_OBSERVATIONS][start_idx:end_idx]

        self.observations = data[c.OBSERVATIONS][start_idx:end_idx]
        self.hidden_states = data[c.HIDDEN_STATES][start_idx:end_idx]
        self.next_observations = data[c.NEXT_OBSERVATIONS][start_idx:end_idx]
        self.next_hidden_states = data[c.NEXT_HIDDEN_STATES][start_idx:end_idx]
        self.actions = data[c.ACTIONS][start_idx:end_idx]
        self.rewards = data[c.REWARDS][start_idx:end_idx]
        self.dones = data[c.DONES][start_idx:end_idx]

        for k in data[c.INFOS]:
            self.infos[k] = data[c.INFOS][k][start_idx:end_idx]

        self._pointer = data[c.POINTER]
        self._count = data[c.COUNT]

        # adjust if sizes change based on different idxes
        if end_idx is not None:
            self._count = min(self._count, end_idx - start_idx)
            self._pointer = self._pointer % len(self)

        self._dtype = data[c.DTYPE]
        if load_rng:
            self.rng = data[c.RNG]

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
    """This buffer stores one trajectory as a sample"""

    def __init__(
        self,
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
        dtype=np.float32,
        policy_switch_discontinuity = False
    ):
        super().__init__(
            memory_size=memory_size,
            obs_dim=obs_dim,
            h_state_dim=h_state_dim,
            action_dim=action_dim,
            reward_dim=reward_dim,
            infos=infos,
            burn_in_window=burn_in_window,
            padding_first=padding_first,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
            rng=rng,
            dtype=dtype,
        )
        self._curr_episode_length = 0
        self._episode_lengths = [0]
        self._episode_start_idxes = [0]
        self._last_observations = []
        self._policy_discont_last_observations = []
        self._policy_lengths = [0]
        self._policy_start_idxes = [0]
        self._curr_active_policy = None
        self._policy_switch_discontinuity = policy_switch_discontinuity

    def push(
        self,
        obs,
        h_state,
        act,
        rew,
        done,
        info,
        next_obs,
        next_h_state,
        **kwargs,
    ):
        if self._policy_switch_discontinuity:
            if self._count == 0:
                self._curr_active_policy = info[c.HIGH_LEVEL_ACTION]
                new_step_policy = None
            else:
                new_step_policy = info[c.HIGH_LEVEL_ACTION]

        self._episode_lengths[-1] += 1

        if self.is_full:
            self._episode_lengths[0] -= 1
            self._episode_start_idxes[0] += 1

            if self._episode_lengths[0] <= 0:
                self._episode_lengths.pop(0)
                self._episode_start_idxes.pop(0)

            if self._policy_switch_discontinuity:
                self._policy_lengths[0] -= 1
                self._policy_start_idxes[0] += 1

                if self._policy_lengths[0] <= 0:
                    self._policy_lengths.pop(0)
                    self._policy_start_idxes.pop(0)

        if type(done) == list and len(done) == 1:
            done = done[0]

        if self._policy_switch_discontinuity and new_step_policy is not None:
            # episode must not be on first step, since that will always erroneously cause another discontinuity,
            # in addition to the one already recorded from done
            if new_step_policy != self._curr_active_policy and self._episode_lengths[-1] > 1:
                self._policy_lengths.append(0)
                # not incremented as in done, since this timestep is new policy
                self._policy_start_idxes.append(self._pointer)
                self._policy_discont_last_observations.append(next_obs)

        if done:
            self._episode_lengths.append(0)
            self._last_observations.append(next_obs)
            self._episode_start_idxes.append(self._pointer + 1)

            # treat done as a policy switch, since they're handled the same way during training
            if self._policy_switch_discontinuity:
                self._policy_lengths[-1] += 1
                self._policy_lengths.append(0)
                self._policy_start_idxes.append(self._pointer + 1)
                self._policy_discont_last_observations.append(next_obs)

        # must be done after policy lengths are appended or not
        if self._policy_switch_discontinuity and not done:
            self._policy_lengths[-1] += 1

        if self._policy_switch_discontinuity and new_step_policy is not None:
            self._curr_active_policy = new_step_policy

        return super().push(
            obs=obs, h_state=h_state, act=act, rew=rew, done=done, info=info
        )

    def sample_trajs(
        self,
        batch_size,
        next_obs,
        idxes=None,
        horizon_length=2,
        **kwargs,
    ):
        raise NotImplementedError("A cleaner, vectorized version of the function here is implemented in torch_pin_buffer. "\
                                  "This is out of date, and not necessary for our experiments, so is not yet fixed.")

        assert horizon_length >= 2, f"horizon_length must be at least length of 2. Got: {horizon_length}"
        if not len(self._episode_lengths):
            raise NoSampleError

        if self._policy_switch_discontinuity:
            discontinuity_lengths = self._policy_lengths
            discontinuity_start_idxes = self._policy_start_idxes
        else:
            discontinuity_lengths = self._episode_lengths
            discontinuity_start_idxes = self._episode_start_idxes

        if idxes is None:
            episode_idxes = self.rng.randint(int(discontinuity_lengths[0] <= 1),
                len(discontinuity_lengths) - int(discontinuity_lengths[-1] <= 1),
                size=batch_size,
            )
        else:
            # episode_idxes is a list of generated/given episode indices, list size is batch_size,
            # these episodes will be presumably used for sampling
            episode_idxes = idxes

        # Get subtrajectory within each episode
        episode_lengths = np.array(discontinuity_lengths)
        episode_start_idxes = np.array(discontinuity_start_idxes)

        # using generated episode indexes above, convert/swap each episode index element into corresponding
        # episode length value, same size as episode_idxes list
        batch_episode_lengths = episode_lengths[episode_idxes]

        sample_lengths = np.tile(np.arange(horizon_length), (batch_size, 1)) #generate stacked lists of arange values up to horizon length
        #generate subtrajectory start indices by sampling random ints, each element of result is a random int with max value of corresponding element in batch_episode_lengths - 1.
        # this presumably randomly generates a starting point/timestep within each episode to sample timesteps from, -1 is to ensure we can always at least sample one timestep before end of episode
        subtraj_start_idxes = self.rng.randint(batch_episode_lengths - 1)
        # episode_start_idxes[episode_idxes] returns the starting timestep index number for each sampled episode index from above, replacing each episode index value with value of its corresponding starting timestep index
        # this is added to the randomly sampled timestep in the middle of each sampled episode given in subtraj_start_idxes
        # [:, None] just converts the list into a 2d list size (batch_size,1)
        # horizon lengths given in sample_lengths(also a 2d list) are added to the 2D list (subtraj_start_idxes + episode_start_idxes[episode_idxes])[:, None]
        #to create a range of n values starting from timestep value computed above, now it is a batch_size x n 2d array where each row is a sequentially incrementing value from the randomly sampled timestep in the middle of episode
        #from left to right
        sample_idxes = (
            (subtraj_start_idxes + episode_start_idxes[episode_idxes])[:, None]
            + sample_lengths
        ) % self._memory_size
        (
            obss,
            h_states,
            acts,
            rews,
            dones,
            infos,
            lengths,
        ) = self.get_transitions(sample_idxes.reshape(-1)) #flatten the above 2d matrix and use to sample transitions/samples

        #calculate the length of subtrajectory taken within each episode then clip the value to max n steps into the future
        # so if we took a subtraj near the start of an episode we only consider n timesteps instead of to the end
        ep_lengths = np.clip(
            batch_episode_lengths - subtraj_start_idxes, a_min=0, a_max=horizon_length
        ).astype(np.int64)

        sample_mask = np.flip(
            np.cumsum(np.eye(horizon_length)[horizon_length - ep_lengths], axis=-1),
            axis=-1,
        )
        sample_idxes = sample_idxes * sample_mask - np.ones(sample_idxes.shape) * (
            1 - sample_mask
        )

        infos[c.EPISODE_IDXES] = episode_idxes

        # If the episode ends too early, then the last observation should be in the trajectory
        # at index length_i of the trajectory.
        for sample_i, (ep_i, length_i) in enumerate(zip(episode_idxes, ep_lengths)):
            if length_i == horizon_length:
                continue

            if self._policy_switch_discontinuity:
                obss[sample_i * horizon_length + length_i] = self._policy_discont_last_observations[ep_i] if ep_i < len(self._policy_discont_last_observations) else next_obs
            else:
                obss[sample_i * horizon_length + length_i] = self._last_observations[ep_i] if ep_i < len(self._last_observations) else next_obs

        return (
            obss,
            h_states,
            acts,
            rews,
            dones,
            infos,
            lengths,
            ep_lengths,
            sample_idxes,
        )
