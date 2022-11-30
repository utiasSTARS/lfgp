import _pickle as pickle
import gzip
import numpy as np
import torch
import os
import argparse

TOP_DIR="/media/starslab/users/trevor-ablett/dac-x/play_xyz/expert-data/"
MID="reset/800_steps-90_sp_point5_play_open_200_extra_lasts/"
END="int_2.gz"
TASKS = ['stack_0', 'bring_0', 'insert_0', 'unstack_stack_env_only_0']
EXPERT_PATH_DICT = {
    "stack_0": os.path.join(TOP_DIR, "open-close-stack-lift-reach-move", MID, END),
    "bring_0": os.path.join(TOP_DIR, "open-close-bring-lift-reach-move", MID, END),
    "insert_0": os.path.join(TOP_DIR, "open-close-insert-bring-lift-reach-move", MID, END),
    "unstack_stack_env_only_0": os.path.join(TOP_DIR, "open-close-unstackstack-lift-reach-move-35M", MID, END),
}

for t in TASKS:
    # src_path = f"data/{t}-expert_data/reset/int_2.gz"
    src_path = EXPERT_PATH_DICT[t]
    dst_path = f"expert-data/{t}/{MID}"
    dst_file = 'int_2.gz'

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

    os.makedirs(dst_path, exist_ok=True)
    torch.save(data, os.path.join(dst_path, dst_file))