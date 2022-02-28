import _pickle as pickle
import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, eval_envs, eval_log_dir, device, eval_i, seed, env_name, render, reward_suc_wrapper=None,
             num_eval_eps=50):

    # obss = [[]]
    # rews = [[]]
    # sucs = [[]]
    # acts = [[]]
    # infos = [[]]
    returns = []
    successes = []
    success = None

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)

    num_eps = 0
    ep_return = 0
    success_latch = False

    while num_eps < num_eval_eps:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs.to(device),
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        if render:
            eval_envs.render()

        # Obser reward and next obs
        prev_obs = obs
        obs, rew, done, info = eval_envs.step(action)

        # fix reward
        if reward_suc_wrapper is not None:
            rew, success = reward_suc_wrapper.get_rew_suc(prev_obs, action, info)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        # obss[-1].append(obs)
        # rews[-1].append(rew)
        # sucs[-1].append(success)
        # acts[-1].append(action)
        # infos[-1].append(info)

        ep_return += rew
        if success:
            success_latch = True

        if done:
            num_eps += 1
            returns.append(ep_return)
            successes.append(int(success_latch))
            ep_return = 0
            success_latch = False

            # obss.append([])
            # rews.append([])
            # sucs.append([])
            # acts.append([])
            # infos.append([])

    # pickle.dump({
    #     "obss": obss,
    #     "rews": rews,
    #     "sucs": sucs,
    #     "acts": acts,
    #     "infos": infos,
    # }, open(f"{env_name}-{seed}-{eval_i}.pkl", "wb"))
    print(" Evaluation using {} episodes: mean reward {:.5f}, suc rate {:.5f} \n".format(
        num_eps, np.mean(returns), np.sum(successes) / len(successes)))

    return returns, successes
