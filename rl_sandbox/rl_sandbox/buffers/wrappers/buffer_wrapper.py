from rl_sandbox.buffers.buffer import Buffer


class BufferWrapper(Buffer):
    def __init__(self, buffer):
        self.buffer = buffer

    def __getattr__(self, attr):
        return getattr(self.buffer, attr)

    def sample(self, batch_size, idxes=None):
        return self.buffer.sample(batch_size, idxes)

    # def sample_with_next_obs(self, batch_size, next_obs, next_h_state=None, idxes=None):
    #     return self.buffer.sample_with_next_obs(batch_size, next_obs, next_h_state, idxes)
    def sample_with_next_obs(self, *args, **kwargs):
        return self.buffer.sample_with_next_obs(*args, **kwargs)

    def sample_consecutive(self, batch_size, end_with_done=False):
        return self.buffer.sample_consecutive(batch_size, end_with_done)

    def sample_init_obs(self, batch_size):
        return self.buffer.sample_init_obs(batch_size)

    def sample_trajs(self, batch_size, next_obs, idxes=None, horizon_length=2):
        return self.buffer.sample_trajs(batch_size, next_obs, idxes, horizon_length)

    @property
    def memory_size(self):
        return self.buffer.memory_size

    @property
    def is_full(self):
        return self.buffer.is_full

    def __len__(self):
        return len(self.buffer)

    def push(self, obs, h_state, act, rew, done, info, *args, **kwargs):
        self.buffer.push(obs, h_state, act, rew, done, info, *args, **kwargs)

    def clear(self):
        return self.buffer.clear()

    def save(self, save_path, **kwargs):
        return self.buffer.save(save_path, **kwargs)

    # def load(self, load_path, load_rng=True):
    #     return self.buffer.load(load_path, load_rng=load_rng)
    def load(self, *args, **kwargs):
        return self.buffer.load(*args, **kwargs)

    def transfer_data(self, load_path):
        return self.buffer.transfer_data(load_path)

    def close(self):
        return self.buffer.close()
