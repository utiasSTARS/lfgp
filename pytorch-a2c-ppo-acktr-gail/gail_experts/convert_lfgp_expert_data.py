import _pickle as pickle
import gzip
import numpy as np
import torch


tasks = ['stack_0', 'bring_0', 'insert_0', 'unstack_stack_env_only_0']

for t in tasks:
    src_path = f"data/{t}-expert_data/reset/int_2.gz"
    dst_path = f"data/{t}-expert_data/reset/int_2.pt"

    src_data = pickle.load(gzip.open(src_path, "rb"))

    print(src_data.keys())

    ep_start_idxes = [0]
    for idx, (curr_obs, next_obs) in enumerate(zip(src_data["observations"][1:], src_data["next_observations"][:-1])):
        if np.any(curr_obs != next_obs):
            ep_start_idxes.append(idx + 1)

    num_eps = len(ep_start_idxes)

    print(num_eps)

    data = {
        'states': src_data["observations"][:, :-1],
        'actions': src_data["actions"],
    }
    torch.save(data, dst_path)