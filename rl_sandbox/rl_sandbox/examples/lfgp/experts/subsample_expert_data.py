"""
Tools for subsampling expert data.
"""

import copy
import numpy as np


def subsample_buffers(buffers, keep_every_nth, keep_first_last=False):
    subsampled_bufs = []
    data_strs = ['observations', 'hidden_states', 'actions', 'rewards', 'dones', 'next_observations',
                 'next_hidden_states']
    for b in buffers:
        initial_offset = np.random.randint(keep_every_nth)
        subsampled_b = copy.deepcopy(b)
        if keep_first_last:
            raise NotImplementedError()
            ends = np.argwhere(np.invert(np.all(b.observations[1:] == b.next_observations[:-1], axis=1)))

        inds = np.array(range(initial_offset, len(b), keep_every_nth))

        for ds in data_strs:
            setattr(subsampled_b, ds, getattr(b, ds)[inds])

        # infos done separately since it's a dict
        for k in b.infos.keys():
            subsampled_b.infos[k] = b.infos[k][inds]

        subsampled_b._pointer = 0
        subsampled_b._count = len(inds)
        subsampled_b._memory_size = len(inds)

        subsampled_bufs.append(subsampled_b)

    return subsampled_bufs