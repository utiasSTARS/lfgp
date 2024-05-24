from functools import partial

import numpy as np
from numpy.linalg import norm

from rl_sandbox.auxiliary_rewards.manipulator_learning.panda.play_xyz_state import AuxiliaryReward


class FromEnvAuxiliaryReward(AuxiliaryReward):
    def __init__(self, env, aux_rewards=()):
        aux_reward_funcs = []
        if aux_rewards == ():
            aux_rewards = env.VALID_AUX_TASKS
        for aux_str in aux_rewards:
            rew_func = partial(env.get_aux_rew, tasks=(aux_str,))
            rew_func.__qualname__ = aux_str
            aux_reward_funcs.append(rew_func)

        super().__init__(aux_reward_funcs, False)
