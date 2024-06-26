import _pickle as pickle
import json
import numpy as np
import os
import timeit
import torch
import glob

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import rl_sandbox.constants as c


def check_load_latest_checkpoint(experiment_config, save_path):
    if experiment_config[c.LOAD_LATEST_CHECKPOINT]:
        paths = glob.glob(os.path.join(save_path, '*'))
        if len(paths) == 0:
            print(f"Warning: load_latest_checkpoint set with no existing experiments at {save_path}, starting new experiment.")
            add_time_tag_to_save_path = True
            experiment_config[c.LOAD_LATEST_CHECKPOINT] = False
        else:
            latest_path = sorted(paths)[-1]
            if not os.path.isfile(os.path.join(latest_path, f"{experiment_config[c.CHECKPOINT_NAME]}_buffer.pkl")):
                print(f"Warning: load_latest_checkpoint set with no existing experiments at {save_path}, starting new experiment.")
                add_time_tag_to_save_path = True
                experiment_config[c.LOAD_LATEST_CHECKPOINT] = False
            else:
                save_path = latest_path
                print(f"Loading latest checkpoint from {save_path}/{experiment_config[c.CHECKPOINT_NAME]}")
                experiment_config[c.BUFFER_SETTING][c.LOAD_BUFFER] = os.path.join(
                    save_path, f"{experiment_config[c.CHECKPOINT_NAME]}_buffer.pkl")
                experiment_config[c.LOAD_MODEL] = os.path.join(
                    save_path, f"{experiment_config[c.CHECKPOINT_NAME]}.pt")
                add_time_tag_to_save_path = False
    else:
        add_time_tag_to_save_path = True

    return save_path, add_time_tag_to_save_path

def check_load_as_jumpoff_point(experiment_config, save_path, add_time_tag_to_save_path):
    if experiment_config.get(c.LOAD_MODEL_NAME, "") != "":
        paths = glob.glob(os.path.join(save_path, '*'))
        if len(paths) == 0:
            raise ValueError(f"No paths found at {save_path} to load jumpoff point from")
        else:
            latest_path = sorted(paths)[-1]
            model_n = experiment_config[c.LOAD_MODEL_NAME]
            buffer_n = experiment_config[c.LOAD_BUFFER_NAME]
            print(f"Loading jumpoff point from {latest_path} with model name {model_n}, buffer name {buffer_n}")

            experiment_config[c.BUFFER_SETTING][c.LOAD_BUFFER] = os.path.join(
                latest_path, f"{buffer_n}_buffer.pkl")
            experiment_config[c.LOAD_MODEL] = os.path.join(
                latest_path, f"{model_n}.pt")
            experiment_config[c.LOAD_TRACKING_DICT] = os.path.join(
                latest_path, f"{model_n}_tracking_dict.pkl")

            save_path = ('/').join(latest_path.split('/')[:-1]) + f'_from_{model_n}'

        add_time_tag_to_save_path = True
    else:
        add_time_tag_to_save_path = add_time_tag_to_save_path

    return save_path, add_time_tag_to_save_path

class DummySummaryWriter():
    def add_scalar(self, arg_1, arg_2, arg_3):
        pass

    def add_scalars(self, arg_1, arg_2, arg_3):
        pass

    def add_text(self, arg_1, arg_2, arg_3):
        pass


def make_summary_writer(save_path, algo, cfg, add_time_tag=True):
    summary_writer = DummySummaryWriter()
    cfg[c.ALGO] = algo
    if save_path is not None:
        if add_time_tag:
            time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
            save_path = f"{save_path}/{time_tag}"
        os.makedirs(save_path, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=f"{save_path}/tensorboard")
        pickle.dump(
            cfg,
            open(f'{save_path}/{algo}_experiment_setting.pkl', 'wb'))
        json.dump(
            cfg,
            open(f'{save_path}/{algo}_experiment_setting.json', 'w'),
            indent=4,
            default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        )

    return summary_writer, save_path

def get_rng_state():
    return {'torch_rng_state': torch.get_rng_state(), 'np_rng_state': np.random.get_state()}

def set_rng_state(torch_rng_state, np_rng_state):
    torch.set_rng_state(torch_rng_state.cpu())  # without .cpu throws a bizarre error about not being a ByteTensor
    np.random.set_state(np_rng_state)

def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, c.MAX_INT)

    np.random.seed(seed)
    torch.manual_seed(seed)


class EpochSummary:
    def __init__(self, default_key_length=10, padding=11):
        self._key_length = default_key_length
        self._padding = padding
        self._summary = dict()
        self._epoch = 0
        self._init_tic = timeit.default_timer()

    def log(self, key, value, track_std=True, track_min_max=True, axis=None):
        self._key_length = max(self._key_length, len(key))
        self._summary.setdefault(key, {
            c.LOG_SETTING: {
                c.STANDARD_DEVIATION: track_std,
                c.MIN_MAX: track_min_max,
                c.AXIS: axis,
            },
            c.CONTENT: []
        })
        self._summary[key][c.CONTENT].append(value)

    def new_epoch(self):
        self._epoch += 1
        self._summary.clear()
        self._curr_tic = timeit.default_timer()

    def print_summary(self):
        toc = timeit.default_timer()
        key_length = self._key_length + self._padding
        print("=" * 100)
        print(f"Epoch: {self._epoch}")
        print(f"Epoch Time Spent: {toc - self._curr_tic}")
        print(f"Total Time Spent: {toc - self._init_tic}")
        print("=" * 100)
        print('|'.join(str(x).ljust(key_length) for x in ("Key", "Content")))
        print("-" * 100)

        # temp fix for scheduler trajs that are not always same length
        if 'update_info/scheduler_traj' in self._summary:
            del self._summary['update_info/scheduler_traj']

        for key in sorted(self._summary):
            val = self._summary[key][c.CONTENT]
            setting = self._summary[key][c.LOG_SETTING]
            try:
                print('|'.join(str(x).ljust(key_length) for x in (f"{key} - AVG", np.mean(val, axis=setting[c.AXIS]))))
                if setting[c.STANDARD_DEVIATION]:
                    print('|'.join(str(x).ljust(key_length) for x in (f"{key} - STD DEV", np.std(val, axis=setting[c.AXIS]))))
                if setting[c.MIN_MAX]:
                    print('|'.join(str(x).ljust(key_length) for x in (f"{key} - MIN", np.min(val, axis=setting[c.AXIS]))))
                    print('|'.join(str(x).ljust(key_length) for x in (f"{key} - MAX", np.max(val, axis=setting[c.AXIS]))))
            except:
                print(val)
                print(key)
                assert 0
        print("=" * 100)
