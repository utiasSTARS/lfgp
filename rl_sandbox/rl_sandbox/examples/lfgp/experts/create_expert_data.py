"""
This script loads up an expert model and generate an expert buffer.

The model path consists of the state dict of the model.

The config path consists of all the settings to load the environment
and preprocessing.

Example usage:
python create_expert_data.py --seed=0 --model_path=./state_dict.pt \
    --config_path=./experiment_setting.pkl --save_path=./expert_buffer.pkl \
        --num_episodes=5 --num_steps=10000
"""

import _pickle as pickle
import argparse
import os
import torch
import numpy as np
from ast import literal_eval

import rl_sandbox.constants as c

from rl_sandbox.agents.hrl_agents import SACXAgent, SACXPlusForcedScheduleAgent
from rl_sandbox.algorithms.sac_x.schedulers import FixedScheduler, UScheduler
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.envs.utils import make_env
from rl_sandbox.envs.wrappers.absorbing_state import check_absorbing
from rl_sandbox.learning_utils import buffer_warmup
from rl_sandbox.examples.lfgp.experts.learning_utils import multi_buffer_warmup
from rl_sandbox.examples.lfgp.experts.scripted_policies import GripperIntentions
from rl_sandbox.model_architectures.utils import make_model, make_optimizer
from rl_sandbox.utils import set_seed
from rl_sandbox.buffers.wrappers.torch_buffer import TorchBuffer


def create_trajectories(args):
    assert args.num_episodes > 0
    assert os.path.isfile(args.model_path)
    assert os.path.isfile(args.config_path)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    set_seed(args.seed)
    with open(args.config_path, "rb") as f:
        config = pickle.load(f)

    env_setting = config[c.ENV_SETTING]
    config[c.DEVICE] = torch.device("cuda:0")
    config[c.INTENTIONS_SETTING][c.KWARGS][c.DEVICE] = config[c.DEVICE]

    # means that expert model not designed for absorbing states
    force_absorbing = not args.no_force_absorbing and not check_absorbing(config)

    # env_setting[c.ENV_WRAPPERS][0][c.KWARGS][c.CREATE_ABSORBING_STATE] = True
    # env_setting[c.ENV_WRAPPERS][0][c.KWARGS][c.MAX_EPISODE_LENGTH] = config['max_episode_length']

    if args.forced_schedule is not None and args.forced_schedule != "":
        forced_schedule = literal_eval(args.forced_schedule)

    # this is a check for forced schedule that use transitions on success
    if args.forced_schedule is not None and args.forced_schedule != "" and \
            forced_schedule[list(forced_schedule.keys())[0]].get('trans_on_suc', False):
        env_setting[c.KWARGS]['sparse_cond_time'] = 0.1  # for faster transitions between intentions
        exp_buffer_inds = []
        sched_inds = []
        for sched_int in forced_schedule.keys():
            sched_inds.append(sched_int)
            for b_ind_list in forced_schedule[sched_int]['buffers']:
                for b_ind in b_ind_list:
                    if b_ind not in exp_buffer_inds: exp_buffer_inds.append(b_ind)

        exp_buffer_inds = sorted(exp_buffer_inds)
        sched_inds = sorted(sched_inds)
        num_buffer_tasks = len(exp_buffer_inds)
        num_sched_tasks = len(sched_inds)

        task_override = sched_inds
        pol_i_to_buffer_i = dict()
        for index, aux in enumerate(exp_buffer_inds):
            pol_i_to_buffer_i[aux] = index

    else:
        if args.aux_override is None or args.aux_override == "":
            num_sched_tasks = config[c.INTENTIONS_SETTING][c.KWARGS][c.TASK_DIM]
            num_buffer_tasks = num_sched_tasks
            task_override = None
            pol_i_to_buffer_i = None
        else:
            task_override = [int(c) for c in args.aux_override.split(',')]
            num_sched_tasks = len(task_override)
            num_buffer_tasks = num_sched_tasks
            pol_i_to_buffer_i = dict()
            for index, aux in enumerate(task_override):
                pol_i_to_buffer_i[aux] = index

    env = make_env(env_setting, seed=args.seed)

    if args.scripted_gripper:
        gripper_dim = config[c.ACTION_DIM] - 1
        intentions = GripperIntentions(config[c.ACTION_DIM], gripper_dim, means=torch.zeros(gripper_dim), vars=torch.ones(gripper_dim))
        args.aux_override = "0,1"
        args.intention_i = None
        args.forced_schedule = None
    else:
        intentions = make_model(config[c.INTENTIONS_SETTING])
        intentions.load_state_dict(
            torch.load(args.model_path, map_location='cuda:' + str(config[c.DEVICE].index))[c.INTENTIONS][c.STATE_DICT])

    config[c.BUFFER_SETTING][c.STORE_NEXT_OBSERVATION] = True
    buffer_preprocessing = config[c.BUFFER_PREPROCESSING]
    config[c.BUFFER_SETTING][c.KWARGS][c.MEMORY_SIZE] = args.num_steps

    # not set up for torch pin buffer so we won't use it
    config[c.BUFFER_SETTING][c.STORAGE_TYPE] = c.RAM
    config[c.BUFFER_SETTING][c.BUFFER_WRAPPERS] = [{c.WRAPPER: TorchBuffer, c.KWARGS: {}},]
    if c.DEVICE in config[c.BUFFER_SETTING][c.KWARGS].keys():
        del config[c.BUFFER_SETTING][c.KWARGS][c.DEVICE]

    scheduler_period = config[c.SCHEDULER_SETTING][c.TRAIN][c.SCHEDULER_PERIOD] \
        if args.scheduler_period is None else args.scheduler_period

    if args.intention_i is None:
        scheduler = UScheduler(num_tasks=num_sched_tasks, task_options=task_override)
        scheduler_period = config[c.SCHEDULER_SETTING][c.TRAIN][c.SCHEDULER_PERIOD] \
            if args.scheduler_period is None else args.scheduler_period
        if force_absorbing: config[c.BUFFER_SETTING][c.KWARGS][c.OBS_DIM] = (config[c.OBS_DIM] + 1,)

        if args.num_steps_per_buffer is None or args.num_steps_per_buffer == "":
            expert_buffers = [make_buffer(config[c.BUFFER_SETTING], args.seed) for _ in range(num_buffer_tasks)]
        else:
            num_steps_per_buffer = [int(c) for c in args.num_steps_per_buffer.split(',')]
            expert_buffers = []
            for i in range(num_buffer_tasks):
                config[c.BUFFER_SETTING][c.KWARGS][c.MEMORY_SIZE] = num_steps_per_buffer[i]
                expert_buffers.append(make_buffer(config[c.BUFFER_SETTING], args.seed))
    else:
        scheduler = FixedScheduler(args.intention_i, num_sched_tasks)
        expert_buffers = [make_buffer(config[c.BUFFER_SETTING], args.seed)]
        pol_i_to_buffer_i = dict()
        pol_i_to_buffer_i[args.intention_i] = 0

    scheduler_args = dict(scheduler=scheduler, intentions=intentions, learning_algorithm=None,
                          scheduler_period=scheduler_period, preprocess=config[c.EVALUATION_PREPROCESSING])
    if args.forced_schedule is None or args.forced_schedule == "":
        agent = SACXAgent(**scheduler_args)
    else:
        agent = SACXPlusForcedScheduleAgent(**scheduler_args, forced_schedule=forced_schedule)

    config[c.NUM_STEPS] = args.num_steps
    config[c.NUM_EPISODES] = args.num_episodes
    if args.num_episodes_per_buffer is not None and args.num_episodes_per_buffer != "":
        config["num_episodes_per_buffer"] = [int(c) for c in args.num_episodes_per_buffer.split(',')]

    def transition_preprocess(obs,
                              h_state,
                              action,
                              reward,
                              done,
                              info,
                              next_obs,
                              next_h_state):

        if force_absorbing:
            obs = np.append(obs, [[0]], axis=-1)
            next_obs = np.append(next_obs, [[0]], axis=-1)
        else:
            if obs[:, -1] == 1:
                action[:] = 0

        return {
            "obs": obs,
            "h_state": h_state,
            "act": action,
            "rew": [reward],
            "done": done,
            "info": info,
            "next_obs": next_obs,
            "next_h_state": next_h_state,
        }

    multi_buffer_warmup(agent=agent,
                        env=env,
                        buffers=expert_buffers,
                        buffer_preprocess=buffer_preprocessing,
                        transition_preprocess=transition_preprocess,
                        experiment_settings=config,
                        reset_between_intentions=args.reset_between_intentions,
                        pol_i_to_buffer_i=pol_i_to_buffer_i,
                        render=args.render,
                        store_success_only=args.success_only,
                        reset_on_success=args.reset_on_success)
    for i, buf in enumerate(expert_buffers):
        buf.save(save_path=args.save_path + 'int_' + str(i) + '.gz', end_with_done=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--no_force_absorbing", default=False, action='store_true', help="Don't force absorbing state if exp model doesn't have.")
    parser.add_argument("--model_path", required=True, type=str, help="The path to load the model")
    parser.add_argument("--config_path", required=True, type=str, help="The path to load the config that trained the model")
    parser.add_argument("--save_path", required=True, type=str, help="The path to save the trajectories")
    parser.add_argument("--num_episodes", required=True, type=int, help="The maximum number of episodes")
    parser.add_argument("--num_steps", required=True, type=int, help="The maximum number of steps per buffer")
    parser.add_argument("--intention_i", required=False, type=int,
                        help="The i'th intention to collect data for -- If not set, collect from all intentions"
                             "uniformly (default)")
    parser.add_argument("--reset_between_intentions", default=False, action="store_true",
                        help="If set, reset environment between intentions.")
    parser.add_argument("--scheduler_period", required=False, type=int,
                        help="Overrides the scheduler period used during training")
    parser.add_argument("--aux_override", required=False, type=str, 
                        help="A string of integers to generate data for, overriding the policy's aux tasks "
                             "with a subset (e.g. \"0, 2, 3, 10\"")
    parser.add_argument("--render", default=False, action="store_true",
                        help="Whether or not to render the data collection")
    parser.add_argument("--success_only", default=False, action="store_true",
                        help="Only save data that achieves success. Must be used with reset_between_intentions.")
    parser.add_argument("--reset_on_success", default=False, action="store_true",
                        help="Reset the env once a policy is successful. Must be used with reset_between_intentions.")
    parser.add_argument("--num_steps_per_buffer", required=False, type=str, help="Max number of steps per buffer,"
                        " if you want each buffer to be a different length. (e.g. \"9000, 9000, 18000, 9000\"")
    parser.add_argument("--forced_schedule", required=False, type=str,
                        help="Forced schedule for specific intentions, see SACXPlusForcedScheduleAgent description "
                             "for details on how to use.")
    parser.add_argument("--num_episodes_per_buffer", required=False, type=str,
                        help="Number of episodes per buffer. If using forced schedule that stores to multiple buffers, "
                             "num episodes per keys in forced schedule. Same format as num_steps_per_buffer.")
    parser.add_argument("--scripted_gripper", action="store_true", 
                        help="Whether or not to use scripted open/close grippers. "
                             "When enabled, will only collect for open/close")
    args = parser.parse_args()

    create_trajectories(args)
