import numpy as np
from numpy.linalg import norm

from rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state import AuxiliaryReward


# reach reward defined based on how defined in original envs combined with how we defined datasets in rce_multitask_envs.py
def reach(observation, next_observation, **kwargs):
    obs = observation
    next_obs = next_observation

    palm_to_ball_dist_before = np.linalg.norm(obs[-9:-6])
    palm_to_ball_dist_after = np.linalg.norm(next_obs[-9:-6])

    return palm_to_ball_dist_before - palm_to_ball_dist_after


def grasp(observation, next_observation, **kwargs):
    obs = observation
    next_obs = next_observation

    ball_target_z_diff_before = abs(obs[-1])
    ball_target_z_diff_after = abs(next_obs[-1])

    return ball_target_z_diff_before - ball_target_z_diff_after


class RelocateHumanV0AuxiliaryReward(AuxiliaryReward):
    def __init__(self, env_name, aux_rewards=('reach',)):
        aux_reward_funcs = [globals()[ar_str] for ar_str in aux_rewards]
        super().__init__(aux_reward_funcs, True)
