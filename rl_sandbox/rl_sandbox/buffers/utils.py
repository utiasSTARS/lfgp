import numpy as np
import gzip
import pickle

import rl_sandbox.constants as c

from rl_sandbox.buffers.disk_buffer import DiskNumPyBuffer
from rl_sandbox.buffers.ram_buffer import NumPyBuffer, NextStateNumPyBuffer, TrajectoryNumPyBuffer
from rl_sandbox.buffers.torch_pin_buffer import TorchPinBuffer, TrajectoryPinBuffer


def make_buffer(buffer_cfg, seed=None, load_buffer=False, start_idx=0, end_idx=None, match_load_size=False,
                frame_stack_load=1):
    if match_load_size and load_buffer:
        with gzip.open(load_buffer, "rb") as f:
            data = pickle.load(f)
        original_size = buffer_cfg[c.KWARGS][c.MEMORY_SIZE]
        buffer_cfg[c.KWARGS][c.MEMORY_SIZE] = data[c.MEMORY_SIZE]

    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    buffer_cfg[c.KWARGS][c.RNG] = np.random.RandomState(seed)

    if buffer_cfg[c.STORAGE_TYPE] == c.DISK:
        buffer = DiskNumPyBuffer(**buffer_cfg[c.KWARGS])
    elif buffer_cfg[c.STORAGE_TYPE] == c.RAM:
        buffer_type = buffer_cfg.get(c.BUFFER_TYPE, c.DEFAULT)
        assert buffer_type in c.VALID_BUFFER_TYPE, f"Invalid buffer type: {buffer_type}"

        # this line for compatibility with old code
        if buffer_cfg.get(c.STORE_NEXT_OBSERVATION, False):
            buffer_type = c.STORE_NEXT_OBSERVATION

        if buffer_type == c.DEFAULT:
            buffer = NumPyBuffer(**buffer_cfg[c.KWARGS])
        elif buffer_type == c.STORE_NEXT_OBSERVATION:
            buffer = NextStateNumPyBuffer(**buffer_cfg[c.KWARGS])
        elif buffer_type == c.TRAJECTORY:
            buffer = TrajectoryNumPyBuffer(**buffer_cfg[c.KWARGS])
        else:
            raise NotImplementedError

    elif buffer_cfg[c.STORAGE_TYPE] == c.GPU:
        buffer = TorchPinBuffer(**buffer_cfg[c.KWARGS])
    elif buffer_cfg[c.STORAGE_TYPE] == c.NSTEP_GPU:
        buffer = TrajectoryPinBuffer(**buffer_cfg[c.KWARGS])
    else:
        raise NotImplementedError

    for wrapper_config in buffer_cfg[c.BUFFER_WRAPPERS]:
        buffer = wrapper_config[c.WRAPPER](buffer, **wrapper_config[c.KWARGS])

    if load_buffer:
        buffer.load(load_buffer, load_rng=seed==None, start_idx=start_idx, end_idx=end_idx, frame_stack=frame_stack_load)
        if match_load_size:
            buffer_cfg[c.KWARGS][c.MEMORY_SIZE] = original_size

    return buffer
