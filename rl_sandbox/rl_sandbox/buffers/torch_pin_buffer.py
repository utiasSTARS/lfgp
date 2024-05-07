from email import policy
import _pickle as pickle
import gzip
import numpy as np
import os
import torch
import time

from collections import deque

from rl_sandbox.buffers.buffer import (
    Buffer,
    CheckpointIndexError,
    LengthMismatchError,
    NoSampleError
)
from rl_sandbox.envs.wrappers.frame_stack import make_frame_stack
import rl_sandbox.constants as c

NATLOG_2 = 0.6931471824645996

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
                 rng=None,
                 policy_switch_discontinuity=False,
                 exponential_sampling_method=None,
                 exponential_sampling_param=0.0,
                 exponential_uniform_prop=0.5):
        self._memory_size = memory_size
        self._dtype = dtype
        self.observations = torch.zeros(size=(memory_size, *obs_dim), dtype=dtype, device=device)
        self.next_observations = torch.zeros(size=(memory_size, *obs_dim), dtype=dtype, device=device)
        self.hidden_states = torch.zeros(size=(memory_size, *h_state_dim), dtype=dtype, device=device)
        self.next_hidden_states = torch.zeros(size=(memory_size, *h_state_dim), dtype=dtype, device=device)
        self.actions = torch.zeros(size=(memory_size, *action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size=(memory_size, *reward_dim), dtype=torch.float32, device=device)
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

        self._exponential_sampling_method = exponential_sampling_method
        self._exponential_sampling_param = exponential_sampling_param
        self._exponential_uniform_prop = exponential_uniform_prop

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

    def _update_exp_dist(self):
        if self._exponential_sampling_method == "med_prop":
            exp_med = self._exponential_sampling_param * len(self)
            rate = torch.tensor(NATLOG_2 / exp_med, device=self.device)
            self._exp_dist = torch.distributions.Exponential(rate=rate)
        elif self._exponential_sampling_method == 'med_fixed':
            if not hasattr(self, '_exp_dist'):
                exp_med = self._exponential_sampling_param
                rate = torch.tensor(NATLOG_2 / exp_med, device=self.device)
                self._exp_dist = torch.distributions.Exponential(rate=rate)
        else:
            raise NotImplementedError(f"not implemented for exponential sampling method {self._exponential_sampling_method}")

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

    def get_transitions_with_next(self, idxes):
        obss, h_states, acts, rews, dones, infos, lengths = self.get_transitions(idxes)
        next_obss = self.next_observations[idxes]
        next_h_states = self.next_hidden_states[idxes]

        return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths

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
            if self._exponential_sampling_method is not None:
                self._update_exp_dist()

                num_exponential = round(self._exponential_uniform_prop * batch_size)
                num_uniform = batch_size - num_exponential

                sample = self._exp_dist.sample((num_exponential,)).type(torch.int64)
                exp_random_idxes = len(self) - sample
                invalid_inds = exp_random_idxes < 0
                if torch.any(invalid_inds):
                    exp_random_idxes[invalid_inds] = torch.randint(len(self), size=(invalid_inds.sum(),), device=self.device)

                uni_random_idxes = torch.randint(len(self), size=(num_uniform,), device=self.device)
                random_idxes = torch.cat([exp_random_idxes, uni_random_idxes])

            else:
                random_idxes = torch.randint(len(self), size=(batch_size,), device=self.device)
        else:
            random_idxes = idxes

        obss, h_states, acts, rews, dones, infos, lengths = self.get_transitions(random_idxes)

        if include_random_idxes:
            return obss, h_states, acts, rews, dones, infos, lengths, random_idxes
        else:
            return obss, h_states, acts, rews, dones, infos, lengths

    def sample_with_next_obs(self, batch_size, next_obs, next_h_state=None, idxes=None, include_random_idxes=False):
        # all TorchPinBuffers are NextStateBuffers, so match NextStateNumPyBuffer
        obss, h_states, acts, rews, dones, infos, lengths, random_idxes = self.sample(batch_size, idxes, True)
        next_obss = self.next_observations[random_idxes][:, None, ...]
        next_h_states = self.next_hidden_states[random_idxes][:, None, ...]

        if include_random_idxes:
            return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, random_idxes
        else:
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
        random_idxes = init_idxes[torch.randint(len(init_idxes), size=(batch_size,), device=self.device)]
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
            random_ending_idx = torch.randint(batch_size, (len(self) + 1,), device=self.device)

        idxes = torch.arange(random_ending_idx - batch_size, random_ending_idx)
        obss, h_states, acts, rews, dones, infos, lengths = self.get_transitions(idxes)

        return obss, h_states, acts, rews, dones, infos, lengths, random_ending_idx

    def save(self, save_path, end_with_done=False):
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

        self.hidden_states = data[c.HIDDEN_STATES][start_idx:end_idx]
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

    def merge(self, other_buf: 'TorchPinBuffer'):
        self._memory_size += other_buf._memory_size
        self._count += other_buf._count
        self._pointer = 0

        for attr in ('observations', 'next_observations', 'hidden_states', 'next_hidden_states', 'actions', 'dones'):
            setattr(self, attr, torch.cat((getattr(self, attr), getattr(other_buf, attr))))

        # handle case where reward dimensions don't match
        cur_rew_dim = self.rewards.shape[-1]
        new_rew_dim = other_buf.rewards.shape[-1]
        if cur_rew_dim != new_rew_dim:
            print("Warning: merging buffers with different rew shape. Rewards may not be compatible.")
        new_rewards = torch.zeros_like(self.rewards)
        max_new_ind = min(cur_rew_dim, new_rew_dim)
        new_rewards[:, :max_new_ind] = other_buf.rewards[:, :max_new_ind]
        self.rewards = torch.cat((self.rewards, new_rewards))

        for k in self.infos:
            self.infos[k] = torch.cat((self.infos[k], other_buf.infos[k]))


class TrajectoryPinBuffer(TorchPinBuffer):
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
        rng=None, #not needed, using torch.randint instead for rng
        dtype=torch.float32,
        device='cpu',
        policy_switch_discontinuity = False,
        n_step=1,
        n_step_to_end=True,
        remove_final_transitions=False,  # old code worked with this, so leaving it as option
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
            device=device,
            rng=rng,
            dtype=dtype,
        )
        self._episode_lengths = [0]
        self._episode_start_idxes = [0]
        self._policy_lengths = [0]
        self._policy_start_idxes = [0]
        self._switch_idxes_tensor = torch.zeros(size=(memory_size,), dtype=torch.int64, device=device)
        self._switch_idxes_tensor_pointer = 1
        self._curr_active_policy = None
        self._policy_switch_discontinuity = policy_switch_discontinuity
        self._valid_base_idxes = torch.zeros(size=(memory_size,), dtype=torch.int64, device=device)
        self._valid_base_idx_pointer = 0
        self._nstep = n_step

        self._n_step_to_end = n_step_to_end
        self._remove_final_transitions = remove_final_transitions

    def load(self, load_path, load_rng=True, start_idx=0, end_idx=None, frame_stack=1):
        super().load(load_path, load_rng, start_idx=start_idx, end_idx=end_idx, frame_stack=frame_stack)

        # now load nstep specific things
        with gzip.open(load_path, "rb") as f:
            data = pickle.load(f)

        if "n_step_data" in data:
            self._episode_lengths = data["episode_lengths"]
            self._episode_start_idxes = data["episode_start_idxes"]
            self._policy_lengths = data["policy_lengths"]
            self._policy_start_idxes = data["policy_start_idxes"]
            self._switch_idxes_tensor = torch.as_tensor(data["switch_idxes_tensor"], device=self.device)
            self._switch_idxes_tensor_pointer = data["switch_idxes_tensor_pointer"]
            self._curr_active_policy = data["curr_active_policy"]
            self._policy_switch_discontinuity = data["policy_switch_discontinuity"]
            self._valid_base_idxes = torch.as_tensor(data["valid_base_idxes"], device=self.device)
            self._valid_base_idx_pointer = data["valid_base_idx_pointer"]
            self._nstep = data["nstep"]
            self._n_step_to_end = data["n_step_to_end"]
            self._remove_final_transitions = data["remove_final_transitions"]
        else:
            # for expert data that isn't stored as trajectory buffers
            # NOTE: this only works if expert data is stored with single task per buffer, as we have been doing

            # get final transition indices of each buffer
            np_obs = self.observations[:len(self)].cpu().numpy()
            np_next_obs = self.next_observations[:len(self)].cpu().numpy()

            ends = np.argwhere(np.invert(np.all(np_obs[1:] == np_next_obs[:-1], axis=1))).flatten()
            ends = np.concatenate([ends, [len(self) - 1]])

            if ends[0] == 0:  # 0 already in
                ends = ends[1:]

            self._switch_idxes_tensor[1:len(ends)+1] = torch.as_tensor(ends, device=self.device)
            self._switch_idxes_tensor_pointer = len(ends) + 1

    def save(self, save_path, end_with_done=False):
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
                c.DTYPE: np.float32,
                "n_step_data": {
                    "episode_lengths": self._episode_lengths,
                    "episode_start_idxes": self._episode_start_idxes,
                    "policy_lengths": self._policy_lengths,
                    "policy_start_idxes": self._policy_start_idxes,
                    "switch_idxes_tensor": self._switch_idxes_tensor.cpu().numpy(),
                    "switch_idxes_tensor_pointer": self._switch_idxes_tensor_pointer,
                    "curr_active_policy": self._curr_active_policy,
                    "policy_switch_discontinuity": self._policy_switch_discontinuity,
                    "valid_base_idxes": self._valid_base_idxes.cpu().numpy(),
                    "valid_base_idx_pointer": self._valid_base_idx_pointer,
                    "nstep": self._nstep,
                    "n_step_to_end": self._n_step_to_end,
                    "remove_final_transitions": self._remove_final_transitions,
                }
            }, f)

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

        policy_switch = False
        if self._policy_switch_discontinuity and new_step_policy is not None:
            # episode must not be on first step, since that will always erroneously cause another discontinuity,
            # in addition to the one already recorded from done
            if new_step_policy != self._curr_active_policy and self._episode_lengths[-1] > 1:
                policy_switch = True
                self._policy_lengths.append(0)
                # not incremented as in done, since this timestep is new policy
                self._policy_start_idxes.append(self._pointer)

        if done:
            self._episode_lengths.append(0)
            self._episode_start_idxes.append(self._pointer + 1)

            # treat done as a policy switch, since they're handled the same way during training
            if self._policy_switch_discontinuity:
                self._policy_lengths[-1] += 1
                self._policy_lengths.append(0)
                self._policy_start_idxes.append(self._pointer + 1)
                policy_switch = True

        # update the switch idxes tensor
        if self._policy_switch_discontinuity and policy_switch:
            self._switch_idxes_tensor[self._switch_idxes_tensor_pointer] = self._policy_start_idxes[-1]
            self._switch_idxes_tensor_pointer += 1
            if self.is_full:
                raise NotImplementedError("Fix this if you need it! Probably need a front pointer too or something.")
        elif not self._policy_switch_discontinuity and done:
            self._switch_idxes_tensor[self._switch_idxes_tensor_pointer] = self._episode_start_idxes[-1]
            self._switch_idxes_tensor_pointer += 1
            if self.is_full:
                raise NotImplementedError("Fix this if you need it! Probably need a front pointer too or something.")

        # must be done after policy lengths are appended or not
        if self._policy_switch_discontinuity and not done:
            self._policy_lengths[-1] += 1

        if self._policy_switch_discontinuity and new_step_policy is not None:
            self._curr_active_policy = new_step_policy

        # check to see if we can add this index to valid sampling indices for n-step
        if self._n_step_to_end:
            # in this case, exclude all policy switch dones or ep dones, and include everything else
            if self._policy_switch_discontinuity:
                if policy_switch and not done:
                    # overwrite previous valid index, since it's the final observation in an episode
                    # and isn't a valid base for n-step
                    self._valid_base_idxes[self._valid_base_idx_pointer - 1] = self._pointer
                valid = not policy_switch and not done
            else:
                valid = not done
            if valid:
                self._valid_base_idxes[self._valid_base_idx_pointer] = self._pointer
                self._valid_base_idx_pointer = (self._valid_base_idx_pointer + 1) % self._memory_size
        else:
            if self._policy_switch_discontinuity:
                ep_i = self._policy_lengths[-1] - 1
            else:
                ep_i = self._episode_lengths[-1] - 1
            if ep_i >= self._nstep:
                # self._valid_base_idxes.append(len(self) - self._nstep)
                self._valid_base_idxes[self._valid_base_idx_pointer] = self._pointer - self._nstep
                self._valid_base_idx_pointer = (self._valid_base_idx_pointer + 1)
                if self.is_full:
                    raise NotImplementedError("Fix this if you need it! Probably need a front pointer too or something.")

        return super().push(
            obs=obs, h_state=h_state, act=act, rew=rew, done=done, info=info, next_obs=next_obs, next_h_state=next_h_state
        )

    def sample_trajs(
        self,
        batch_size,
        next_obs,
        idxes=None,
        horizon_length=2,
        **kwargs,
    ):

        if idxes is not None:
            assert len(idxes) == batch_size
            sample_idxes = idxes
        else:
            if self._n_step_to_end and not self._remove_final_transitions:
                # can sample from entire buffer since we'll replace any that are less than n away from a transition
                # with repeats
                if self._exponential_sampling_method is not None:
                    self._update_exp_dist()

                    num_exponential = round(self._exponential_uniform_prop * batch_size)
                    num_uniform = batch_size - num_exponential

                    sample = self._exp_dist.sample((num_exponential,)).type(torch.int64)
                    exp_random_idxes = len(self) - sample
                    invalid_inds = exp_random_idxes < 0
                    if torch.any(invalid_inds):
                        exp_random_idxes[invalid_inds] = torch.randint(len(self), size=(invalid_inds.sum(),), device=self.device)

                    uni_random_idxes = torch.randint(len(self), size=(num_uniform,), device=self.device)
                    sample_idxes = torch.cat([exp_random_idxes, uni_random_idxes])

                else:
                    sample_idxes = torch.randint(len(self), (batch_size,), device=self.device)
            else:
                # randomly sample from valid sample indices only
                assert self._exponential_sampling_med_mult == 0, "Implement exponential dist here if you need it"
                sample_options = self._valid_base_idxes[:self._valid_base_idx_pointer - 1]  # don't take current obs
                raw_sample_idxes = torch.randint(sample_options.shape[0], (batch_size,), device=self.device)
                sample_idxes = sample_options[raw_sample_idxes]

        # make sample indices include up to nstep
        sample_lengths = torch.arange(self._nstep + 1, device=self.device).repeat(batch_size, 1)
        sample_idxes_nstep = sample_idxes[:, None] + sample_lengths

        # ensure we don't error on small buffers
        sample_idxes_nstep = torch.minimum(sample_idxes_nstep, torch.tensor(len(self) - 1))

        # we can get transitions in the shape that we want, which is (batch_size, nstep+1)
        # obss, h_states, acts, rews, dones, infos, lengths = self.get_transitions(sample_idxes_nstep)
        obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths = \
            self.get_transitions_with_next(sample_idxes_nstep)
        obss = obss.squeeze()
        h_states = h_states.squeeze()
        next_obss = next_obss.squeeze()
        next_h_states = next_h_states.squeeze()

        if self._n_step_to_end:
            # replace obss/acts when we go past the end of transition/episode
            # with just repeats of the last obs/act pair (i.e. before the policy switch)

            switch_idxes = self._switch_idxes_tensor[:self._switch_idxes_tensor_pointer]
            episode_end_idxes = switch_idxes[1:]
            episode_end_idxes = torch.cat((episode_end_idxes, torch.tensor([len(self)], device=self.device)))
            batch_episode_idxes = torch.searchsorted(switch_idxes, sample_idxes, right=True) - 1
            batch_end_idxes = episode_end_idxes[batch_episode_idxes]
            batch_steps_from_end = batch_end_idxes - sample_idxes - 1

            end_idx = torch.minimum(batch_steps_from_end, torch.tensor(self._nstep, device=self.device))

            end_mask = torch.zeros(obss.shape, dtype=torch.bool, device=self.device)
            end_mask[torch.arange(batch_size), end_idx, :] = True
            end_obss = obss[end_mask].reshape(batch_size, -1)
            end_next_obss = next_obss[end_mask].reshape(batch_size, -1)

            a_end_mask = torch.zeros(acts.shape, dtype=torch.bool, device=self.device)
            a_end_mask[torch.arange(batch_size), end_idx, :] = True
            end_acts = acts[a_end_mask].reshape(batch_size, -1)

            replace_mask = (torch.arange(self._nstep + 1, device=self.device)[:, None] >= end_idx).T
            replace_mask = replace_mask[:, :, None]
            o_replace_mask = replace_mask.expand(-1, -1, obss.shape[-1])
            a_replace_mask = replace_mask.expand(-1, -1, acts.shape[-1])

            # vectorized way to replace all obs/acts using broadcasting
            next_obss = torch.where(o_replace_mask, end_next_obss[:, None, :], next_obss[:, :, :])
            obss = torch.where(o_replace_mask, end_obss[:, None, :], obss[:, :, :])
            acts = torch.where(a_replace_mask, end_acts[:, None, :], acts[:, :, :])

            ep_lengths = end_idx

        # return obss, h_states, acts, rews, dones, infos, lengths, sample_idxes, ep_lengths
        return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, sample_idxes, ep_lengths

    def merge(self, other_buf: 'TrajectoryPinBuffer'):
        print("Warning: merge might not work properly for n-step/TrajectoryPinBuffer, be careful!")
        super().merge(other_buf)