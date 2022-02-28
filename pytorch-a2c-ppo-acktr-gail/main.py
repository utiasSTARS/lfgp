import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from datetime import datetime
import pickle
import shutil
from functools import partial
from rl_sandbox.envs.utils import make_env
import rl_sandbox.constants as c
import rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state as p_aux

from gym.spaces.box import Box

class DummyEnvWrapper():
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        return torch.tensor([self.env.reset()], dtype=torch.float)

    def step(self, action):
        obs, rew, done, info = self.env.step(action[0].cpu().numpy())

        if done:  # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py
            info["terminal_observation"] = obs
            obs = self.env.reset()

        return torch.tensor([obs], dtype=torch.float), torch.tensor([rew], dtype=torch.float), torch.tensor([done]), [info]

    def render(self):
        return self.env.render()


class LfGPRewardWrapper():
    """ For getting the rewards that we used in the LfGP paper, which aren't actually part of the env itself. """
    def __init__(self, env, main_task):
        aux_reward_all = p_aux.PandaPlayXYZStateAuxiliaryReward(main_task, include_main=False)

        if main_task == 'unstack_stack_env_only_0':
            main_task = 'stack_0'

        aux_reward_i = aux_reward_all._aux_rewards_str.index(main_task)
        self.reward_func = aux_reward_all._aux_rewards[aux_reward_i]
        self.success_func = partial(env.env.get_task_successes, tasks=[main_task])

    def get_reward(self, prev_obs, act, infos):
        return self.reward_func(info={"infos": infos},
                                observation=prev_obs.cpu().numpy(),
                                action=act[0].cpu().numpy())

    def get_success(self, prev_obs, act, infos):
        return int(self.success_func(observation=prev_obs.cpu().numpy(),
                                     action=act[0].cpu().numpy(),
                                     env_info=infos[-1])[0])
    def get_rew_suc(self, prev_obs, act, infos):
        return self.get_reward(prev_obs, act, infos), self.get_success(prev_obs, act, infos)

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
    eval_log_dir = f"eval_logs/{args.env_name}/{args.seed}/gail/test0/{time_tag}"
    eval_log_file = eval_log_dir + "/train.pkl"  # matches lfgp
    os.makedirs(eval_log_dir)

    eval_dict = {"evaluation_returns": [], "evaluation_successes": [], "evaluation_successes_all_tasks": []}  # matches lfgp stuff
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env_config = {
        c.ENV_BASE: {
            c.ENV_NAME: "PandaPlayInsertTrayXYZState",
        },
        c.KWARGS: {
            c.MAIN_TASK: args.env_name,
            "egl": False,  # necessary for CC, useful for other stuff since we don't need it, but slows down rendering
        },
        c.ENV_TYPE: c.MANIPULATOR_LEARNING,
        c.ENV_WRAPPERS: [],
    }

    envs = DummyEnvWrapper(make_env(env_config, seed=args.seed))
    envs.observation_space = Box(np.array([-np.inf for _ in range(59)]), np.array([np.inf for _ in range(59)]))
    envs.action_space = Box(np.array([-1 for _ in range(4)]), np.array([1 for _ in range(4)]))
    eval_envs = DummyEnvWrapper(make_env(env_config, seed=args.seed + 1))
    eval_envs.observation_space = Box(np.array([-np.inf for _ in range(59)]), np.array([np.inf for _ in range(59)]))
    eval_envs.action_space = Box(np.array([-1 for _ in range(4)]), np.array([1 for _ in range(4)]))

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                      args.gamma, args.log_dir, device, False)

    reward_suc_wrapper = LfGPRewardWrapper(envs, args.env_name)
    eval_reward_suc_wrapper = LfGPRewardWrapper(eval_envs, args.env_name)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(args.gail_experts_file)
        
        # expert_dataset = gail.ExpertDataset(
        #     file_name, num_trajectories=4, subsample_frequency=20)

        expert_dataset = gail.LfGPExpertDataset(
            file_name)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    print(envs.observation_space.shape)
    print(rollouts.obs.shape)
    obs = torch.tensor(envs.reset(), device=device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_successes = deque(maxlen=10)
    episode_return = 0
    ep_success_latch = False

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    eval_i = 0
    for j in range(num_updates + 1):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            prev_obs = obs
            obs, reward, done, infos = envs.step(action)
            obs = torch.tensor(obs, device=device)

            # fix reward
            reward, success = reward_suc_wrapper.get_rew_suc(prev_obs, action, infos)
            reward = torch.tensor(reward)

            if success:
                ep_success_latch = True

            if args.train_render:
                envs.render()

            episode_return += np.array(reward).item()

            # doesn't apply to our panda play environments
            # for info in infos:
            #     if 'episode' in info.keys():
            #         episode_rewards.append(info['episode']['r'])

            if done:
                episode_rewards.append(episode_return)
                episode_successes.append(int(ep_success_latch))
                ep_success_latch = False
                episode_return = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

            if (args.eval_interval is not None and (j * args.num_steps + step + 1) % args.eval_interval == 0):
                # obs_rms = utils.get_vec_normalize(envs).obs_rms
                returns, successes = evaluate(actor_critic, eval_envs, eval_log_dir, device, eval_i, args.seed,
                                              args.env_name, args.eval_render,
                                              eval_reward_suc_wrapper, args.eval_eps)
                eval_dict["evaluation_returns"].append(returns)
                eval_dict["evaluation_successes"].append(successes)
                eval_dict["evaluation_successes_all_tasks"].append(successes)
                if os.path.exists(eval_log_file):
                    shutil.copy(eval_log_file, eval_log_file + ".bkup")
                pickle.dump(eval_dict, open(eval_log_file, 'wb'))
                torch.save(actor_critic, os.path.join(eval_log_dir, f"{j * args.num_steps + step}.pt"))
                if os.path.exists(eval_log_file + ".bkup"):
                    os.remove(eval_log_file + ".bkup")
                eval_i += 1

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            # if j >= 10:
            #     envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #         or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass

        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, suc rate {:.1f}\n"
                .format(j, total_num_steps,
                        int(args.num_steps * args.log_interval / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.sum(episode_successes) / len(episode_successes),
                        dist_entropy, value_loss,  # these not included
                        action_loss))
            start = end


if __name__ == "__main__":
    main()
