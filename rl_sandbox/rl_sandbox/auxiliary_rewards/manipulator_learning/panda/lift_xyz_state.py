import numpy as np

from numpy.linalg import norm


TABLE_HEIGHT = 0.6247  # not definitively defined anywhere, just found through trial and error
BLOCK_HEIGHT_ON_TABLE = 0.6550  # again, trial and error

# taken directly from sac-x paper
def opened(info, **kwargs):
    return 1 if np.all(info["infos"][-1]['grip_pos'] >= .9) else 0

def closed(info, **kwargs):
    return 1 if np.all(info["infos"][-1]['grip_pos'] <= .1) else 0

def lifted(info, max_rew_height=.1, **kwargs):
    block_height = info["infos"][-1]['obj_pos_world'][0][2] - BLOCK_HEIGHT_ON_TABLE
    if block_height > max_rew_height:
        return 1.5
    elif block_height < .005:
        return 0
    else:
        return block_height / max_rew_height

# this is just the generic one, not meant to be used on its own as an aux reward
def close(dist_thresh, obj_1_pos, obj_2_pos, tanh_multiplier=10.0, close_rew=1.5):
    dist = norm(obj_1_pos - obj_2_pos)
    if dist < dist_thresh:
        return close_rew
    else:
        # return 1 - (np.tanh(dist / 10))**2  # from SAC-X paper, but very poorly scaled for meters as units
        return 1 - np.tanh(tanh_multiplier * dist)

def hand_block_close(info, **kwargs):
    return close(0.0, info["infos"][-1]['obj_pos'][:3], info["infos"][-1]['pos'])  # only for first aka blue block

# modified rewards to make more "human like" intentions
def open_action(action, **kwargs):
    action_mag = norm(action[:3])
    open_rew = 1 if action[-1] < 0 else 0
    return open_rew - .5 * action_mag

def close_action(action, **kwargs):
    action_mag = norm(action[:3])
    close_rew = 1 if action[-1] > 0 else 0
    return close_rew - .5 * action_mag

def hand_block_close_speed_penalty(info, action, **kwargs):
    close_rew = hand_block_close(info, **kwargs)
    dist = norm(info["infos"][-1]['obj_pos'][:3] - info["infos"][-1]['pos'])
    action_mag = norm(action[:3])
    speed_penalty = (1. - np.tanh(10 * dist)) * action_mag
    return close_rew


class PandaLiftXYZStateAuxiliaryReward:
    def __init__(self, aux_rewards=(open_action, close_action, lifted, hand_block_close_speed_penalty), include_main=True):
        self._aux_rewards = aux_rewards
        self._include_main = include_main

        # self._done_failure_reward = -5
        # self._done_success_reward = 100

    @property
    def num_auxiliary_rewards(self):
        return len(self._aux_rewards)

    def reward(self,
               observation,
               action,
               reward,
               done,
               next_observation,
               info):
        observation = observation.reshape(-1)
        next_observation = next_observation.reshape(-1)
        reward_vector = []
        if self._include_main:
            reward_vector.append(reward)
        for task_reward in self._aux_rewards:
            reward_vector.append(task_reward(observation=observation,
                                             action=action,
                                             reward=reward,
                                             next_observation=next_observation,
                                             done=done,
                                             info=info))

        return np.array(reward_vector, dtype=np.float32)
