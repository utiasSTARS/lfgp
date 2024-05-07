import numpy as np
from numpy.linalg import norm

from rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state import AuxiliaryReward


# reach reward defined based on how defined in original envs combined with how we defined datasets in rce_multitask_envs.py
def reach(observation, next_observation, **kwargs):
    obs = observation
    next_obs = next_observation

    palm_to_handle_dist_before = np.linalg.norm(obs[-4:-1])
    palm_to_handle_dist_after = np.linalg.norm(next_obs[-4:-1])

    return palm_to_handle_dist_before - palm_to_handle_dist_after


def grasp(observation, next_observation, **kwargs):
    obs = observation
    next_obs = next_observation

    latch_turn_before = obs[27]
    latch_turn_after = next_obs[27]

    return latch_turn_after - latch_turn_before


class DoorHumanV0AuxiliaryReward(AuxiliaryReward):
    def __init__(self, env_name, aux_rewards=('reach',)):
        aux_reward_funcs = [globals()[ar_str] for ar_str in aux_rewards]
        super().__init__(aux_reward_funcs, True)
