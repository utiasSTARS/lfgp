import numpy as np
from numpy.linalg import norm

from rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state import AuxiliaryReward


# reach reward defined based on how rewards are defined for rce envs
def reach(observation, next_observation, **kwargs):
    obs = observation
    next_obs = next_observation

    ee_pos = obs[:3]
    obj_pos = obs[3:6]
    next_ee_pos = next_obs[:3]
    next_obj_pos = next_obs[3:6]

    d_before = norm(ee_pos - obj_pos)
    d_after = norm(next_ee_pos - next_obj_pos)
    return d_before - d_after


def grasp(observation, next_observation, action, **kwargs):
    obs = observation
    next_obs = next_observation

    reach_rew = reach(obs, next_obs)

    obj_z_pos = obs[5]
    next_obj_z_pos = next_obs[5]
    z_inc = next_obj_z_pos - obj_z_pos
    grip_pos = obs[6:8]
    next_grip_pos = next_obs[6:8]
    grip_inc = next_grip_pos[0] - grip_pos[0] - (next_grip_pos[1] - grip_pos[1])  # 2nd index goes negative as it closes

    grasp_rew = z_inc + grip_inc

    return reach_rew + grasp_rew

class SawyerAuxiliaryReward(AuxiliaryReward):
    def __init__(self, env_name, aux_rewards=('reach',)):
        aux_reward_funcs = [globals()[ar_str] for ar_str in aux_rewards]
        super().__init__(aux_reward_funcs, True)
