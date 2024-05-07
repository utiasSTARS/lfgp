import numpy as np
import os

os.environ["MUJOCO_GL"] = "egl"

import rl_sandbox.constants as c
from rl_sandbox.envs.wrappers.absorbing_state import AbsorbingStateWrapper

def make_env(env_config, seed=None):
    assert env_config[c.ENV_TYPE] in c.VALID_ENV_TYPE
    if env_config[c.ENV_TYPE] == c.GYM:
        import gym
        import pybullet_envs
        env = gym.make(**env_config[c.ENV_BASE])
    elif env_config[c.ENV_TYPE] == c.DM_CONTROL:
        from dm_control import suite
        env = suite.load(**env_config[c.ENV_BASE])
    elif env_config[c.ENV_TYPE] == c.MANIPULATOR_LEARNING:
        import manipulator_learning.sim.envs as manlearn_envs
        env = getattr(manlearn_envs,
                      env_config[c.ENV_BASE][c.ENV_NAME])(dense_reward=False, **env_config.get(c.KWARGS, {}))
    elif env_config[c.ENV_TYPE] in [c.SAWYER, c.HAND_DAPG]:
        import rl_sandbox.envs.rce_envs as rce_envs
        env = rce_envs.load_env(env_config[c.ENV_BASE][c.ENV_NAME], gym_env=True, **env_config.get(c.KWARGS, {}))
    else:
        raise NotImplementedError

    for wrapper_config in env_config[c.ENV_WRAPPERS]:
        env = wrapper_config[c.WRAPPER](env, **wrapper_config[c.KWARGS])

    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    env.seed(seed)

    return env


def absorbing_check(algo_params):
    absorbing_in_settings = False
    if c.ENV_WRAPPERS in algo_params[c.ENV_SETTING]:
        for wrapper in algo_params[c.ENV_SETTING][c.ENV_WRAPPERS]:
            if wrapper[c.WRAPPER] == AbsorbingStateWrapper:
                absorbing_in_settings = True

    return absorbing_in_settings