"""
This script loads up a trained model and performs evaluation using the model.
During evaluation, we use deterministic action (usually the mean of the action).

The model path consists of the state dict of the model.

The config path consists of all the settings to load the environment
and preprocessing.

Example usage:
python evaluate.py --seed=0 --model_path=./state_dict.pt \
    --config_path=./experiment_setting.pkl --num_episodes=5
"""

import _pickle as pickle
import argparse
import numpy as np
import torch
import os
from functools import partial
from ast import literal_eval

import rl_sandbox.constants as c

from rl_sandbox.examples.eval_tools.utils import load_model
from rl_sandbox.learning_utils import evaluate_policy
from rl_sandbox.utils import set_seed


def evaluate(args):
    assert args.num_episodes > 0

    # need to initially load to get number of intentions in model
    config, env, buffer_preprocessing, agent = load_model(args.seed, args.config_path, args.model_path, args.intention,
                                                          args.device, include_disc=False)
    
    if args.intention == -1:
        eval_intentions = list(range(config[c.NUM_TASKS]))
    else:
        eval_intentions = [args.intention]
    
    all_int_ret = []
    all_int_suc = []
    exe_int_ret = []
    exe_int_suc = []
    for eval_intention in eval_intentions:
        set_seed(args.seed)
        config, buffer_preprocessing, agent = load_model(args.seed, args.config_path, args.model_path, eval_intention,
                                                         args.device, include_env=False, include_disc=False)
        
        # load up aux rewards and aux successes
        if c.AUXILIARY_REWARDS in config:
            auxiliary_reward = config[c.AUXILIARY_REWARDS].reward
            if hasattr(config[c.AUXILIARY_REWARDS], 'set_aux_rewards_str'):
                config[c.AUXILIARY_REWARDS].set_aux_rewards_str()
        elif c.EVALUATION_REWARD_FUNC in config:  # handles BC and DAC
            auxiliary_reward = config[c.EVALUATION_REWARD_FUNC]
        else:
            auxiliary_reward = lambda reward, **kwargs: np.array([reward])

        if auxiliary_reward is None:
            auxiliary_reward = lambda reward, **kwargs: np.array([reward])

        if hasattr(env, 'get_task_successes') and c.AUXILIARY_REWARDS in config and \
                hasattr(config[c.AUXILIARY_REWARDS], '_aux_rewards_str'):
            auxiliary_success = partial(
                env.get_task_successes, tasks=config[c.AUXILIARY_REWARDS]._aux_rewards_str)
        elif hasattr(env, 'get_task_successes') and hasattr(env, 'VALID_AUX_TASKS') and \
                (auxiliary_reward.__qualname__ in env.VALID_AUX_TASKS or
                 auxiliary_reward.__qualname__ == 'evaluate.<locals>.<lambda>'):
            if auxiliary_reward.__qualname__ == 'evaluate.<locals>.<lambda>':
                auxiliary_success = partial(env.get_task_successes, tasks=['main'])
            else:
                auxiliary_success = partial(env.get_task_successes, tasks=[auxiliary_reward.__qualname__])
        else:
            auxiliary_success = None

        forced_schedule = None if args.forced_schedule == "" else literal_eval(args.forced_schedule)

        print(f"Running evaluation for intention {eval_intention}")
        rets, _, all_suc = evaluate_policy(agent=agent,
                                           env=env,
                                           buffer_preprocess=buffer_preprocessing,
                                           num_episodes=args.num_episodes,
                                           clip_action=config[c.CLIP_ACTION],
                                           min_action=config[c.MIN_ACTION],
                                           max_action=config[c.MAX_ACTION],
                                           render=args.render,
                                           auxiliary_reward=auxiliary_reward,
                                           auxiliary_success=auxiliary_success,
                                           verbose=True,
                                           forced_schedule=forced_schedule,
                                           stochastic_policy=args.stochastic)

        print("=" * 100)
        print("Interacted with {} episodes".format(args.num_episodes))
        print("Average Return: {} - Std: {}".format(np.mean(rets, axis=1), np.std(rets, axis=1)))
        print("Average Success: {} - Std: {}".format(np.mean(all_suc, axis=1), np.std(all_suc, axis=1)))

        # all_int_ret.extend(rets.T)  # this way matches RecycleScheduler from other code
        # all_int_suc.extend(all_suc.T)

        all_int_ret.append(rets)
        all_int_suc.append(all_suc)

        # extract the executed tasks, so we don't have to do it in other code
        exe_int_ret.append(rets[eval_intention])
        exe_int_suc.append(all_suc[eval_intention])
    
    all_int_ret = np.array(all_int_ret).squeeze()
    all_int_suc = np.array(all_int_suc).squeeze()
    exe_int_ret = np.array(exe_int_ret).squeeze()
    exe_int_suc = np.array(exe_int_suc).squeeze()

    if args.save_path == "":
        model_name = os.path.basename(args.model_path).split('.')[0]
        save_path = os.path.dirname(args.model_path) + f"/eval_{model_name}_{args.num_episodes}_eps_per_int.pkl"
    else:
        save_path = args.save_path
    print(f"Saving model to {save_path}")
    pickle.dump({'evaluation_returns': all_int_ret, 'evaluation_successes_all_tasks': all_int_suc, 
                 'executed_task_returns': exe_int_ret, 'executed_task_successes': exe_int_suc},
                open(f'{save_path}', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--model_path", required=True, type=str, help="The path to load the model")
    parser.add_argument("--config_path", required=True, type=str, help="The path to load the config that trained the model")
    parser.add_argument("--num_episodes", required=True, type=int, help="The number of episodes (per intention)")
    parser.add_argument("--intention", type=int, default=0, help="The intention to use for SAC-X. If set to -1, all"
                                                                 " intentions will be evalutated.")
    parser.add_argument("--render", action="store_true", help="Whether or not to render")
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    parser.add_argument("--stochastic", action="store_true", help="Whether to use stochastic policy")
    parser.add_argument("--save_path", required=False, type=str, default="",
                        help="Path to save results to. Defaults to model path.")
    parser.add_argument("--forced_schedule", required=False, type=str, default="",
                        help="Forced schedule for hierarchical agent in dict form, e.g. {0: 2, 90: 0}")
    args = parser.parse_args()

    evaluate(args)
