import numpy as np
from numpy.linalg import norm

from rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state import AuxiliaryReward


# reach reward defined based on how defined in original envs combined with how we defined datasets in rce_multitask_envs.py
def reach(observation, next_observation, **kwargs):
    obs = observation
    next_obs = next_observation

    palm_to_hammer_dist_before = np.linalg.norm(np.array(obs[-13:-10])-np.array(obs[-10:-7]))
    palm_to_hammer_dist_after = np.linalg.norm(np.array(next_obs[-13:-10])-np.array(next_obs[-10:-7]))

    return palm_to_hammer_dist_before - palm_to_hammer_dist_after


def grasp(observation, next_observation, **kwargs):
    obs = observation
    next_obs = next_observation

    hammer_height_before = obs[-8]
    hammer_height_after = next_obs[-8]

    return hammer_height_after - hammer_height_before


class HammerHumanV0AuxiliaryReward(AuxiliaryReward):
    def __init__(self, env_name, aux_rewards=('reach',)):
        aux_reward_funcs = [globals()[ar_str] for ar_str in aux_rewards]
        super().__init__(aux_reward_funcs, True)
