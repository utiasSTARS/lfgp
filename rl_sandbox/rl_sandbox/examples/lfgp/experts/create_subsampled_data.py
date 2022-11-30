"""
This script loads up existing buffers and generates subsampled versions.

Compared with subsampling on the fly, this ensures that all methods use the exact same data.

Example usage:
python create_subsampled_data.py --seed=0 --input_path=./expert_data \
    --output_path=./expert_data_subsampled --keep_every_nth=20
"""

import copy
import glob
import gzip
import _pickle as pickle
import argparse
import os
import numpy as np

import rl_sandbox.constants as c


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="The random seed")
parser.add_argument("--input_path", required=True, type=str, help="The path to .gz file(s) with expert data")
parser.add_argument("--output_path", required=True, type=str, help="The path to save the new data")
parser.add_argument("--keep_every_nth", required=False, type=int, help="Keep every nth piece of data.")
parser.add_argument("--keep_first_last", action='store_true', help="Keep the first and last of each trajectory, "\
                                                                   "and otherwise subsample normally.")
parser.add_argument("--keep_last", action='store_true', help="Keep the last of each trajectory, "\
                                                             "and otherwise subsample normally.")
parser.add_argument("--keep_last_only", action='store_true', help="Keep the last of each trajectory exclusively.")
parser.add_argument("--num_to_keep", required=False, type=int, help="Cap the amount of data to keep in each intention.")
parser.add_argument("--num_extra_lasts", required=False, type=int, help="Add a number of extra final transitions to each intention.")
parser.add_argument("--num_to_subsample_from", required=False, type=int, help="Cap amount of data to subsample from, but firsts/lasts can still come from more.")

args = parser.parse_args()


SUBSAMPLE_KEYS = ['observations', 'hidden_states', 'actions', 'rewards', 'dones', 'next_observations',
                  'next_hidden_states']

np.random.seed(args.seed)
data_paths = glob.glob(os.path.join(args.input_path, '*.gz'))

assert(os.path.exists(args.input_path)), f"No data folder found at {args.input_path}"
assert sum([args.keep_first_last, args.keep_last, args.keep_last_only]) <= 1, "Can only set one of these."
# assert not (args.keep_first_last and args.keep_last), "Can't set both keep_first_last and keep_last"

if os.path.exists(args.output_path):
    overwrite = input("Output path already exists. Overwrite? Anything but \"yes\" exits.")
    if overwrite != 'yes':
        exit(0)

os.makedirs(args.output_path, exist_ok=True)

for dp in data_paths:
    gz_filename = dp.split('/')[-1]
    out_path = os.path.join(args.output_path, gz_filename)

    with gzip.open(dp, 'rb') as f:
        data = pickle.load(f)

    out_data = copy.deepcopy(data)

    if args.keep_first_last or args.keep_last or args.keep_last_only:
        inds = []
        ends = np.argwhere(np.invert(np.all(data['observations'][1:] == data['next_observations'][:-1], axis=1)))
        starts = np.concatenate([[[0]], ends + 1])

        if args.keep_last_only:
            inds = ends
        else:
            for start, end in zip(starts, ends):
                if args.keep_first_last:
                    inds.append(int(start))
                if end == start:  # should only happen if very first index is an end
                    if args.keep_last:
                        inds.append(int(end))
                    continue

                initial_offset = np.random.randint(args.keep_every_nth)
                next_i = start + initial_offset
                while next_i < end:
                    inds.append(int(next_i))
                    next_i += args.keep_every_nth

                inds.append(int(end))

        inds = np.array(inds).squeeze()

        if args.num_to_subsample_from is not None:
            inds = inds[inds < args.num_to_subsample_from]
    else:
        initial_offset = np.random.randint(args.keep_every_nth)

        if args.num_to_subsample_from is None:
            max_ind = len(data['observations'])
        else:
            max_ind = args.num_to_subsample_from

        # this assumes that the buffers are coming in as only being the size that they need to be
        inds = np.array(range(initial_offset, max_ind, args.keep_every_nth))

    if args.num_to_keep is not None:
        assert len(inds) >= args.num_to_keep, f"Not enough timesteps, wanted {args.num_to_keep}, found "\
                                                f"{len(inds)} for {gz_filename}."
        inds = inds[:args.num_to_keep]

    if args.num_extra_lasts is not None:
        ends = np.argwhere(np.invert(np.all(data['observations'][1:] == data['next_observations'][:-1], axis=1)))
        unused_ends = ends[ends > inds[-1]]
        if unused_ends.shape[0] < args.num_extra_lasts:
            print(f"WARNING: wanted {args.num_extra_lasts} extra lasts, but only found {unused_ends.shape[0]} for {gz_filename}")
        inds = np.concatenate([inds, unused_ends[:args.num_extra_lasts]])

    print(f"Keeping {len(inds)} data for {gz_filename}.")

    for k in SUBSAMPLE_KEYS:
        out_data[k] = data[k][inds]

    for ik in data['infos'].keys():
        out_data['infos'][ik] = data['infos'][ik][inds]

    # also need to update size parameters
    out_data['pointer'] = 0
    out_data['count'] = len(inds)
    out_data['memory_size'] = len(inds)

    with gzip.open(out_path, "wb") as f:
        pickle.dump(out_data, f)

print(f"Subsampled data created and saved to {args.output_path}.")