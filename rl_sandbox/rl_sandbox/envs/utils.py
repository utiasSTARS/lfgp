import numpy as np
import os

os.environ["MUJOCO_GL"] = "egl"

import rl_sandbox.constants as c

def make_env(env_config, seed=None):
    assert env_config[c.ENV_TYPE] in c.VALID_ENV_TYPE
    if env_config[c.ENV_TYPE] == c.GYM:
        import gym
        import pybullet_envs
        env = gym.make(**env_config[c.ENV_BASE])
    elif env_config[c.ENV_TYPE] == c.DM_CONTROL:
        from dm_control import suite
        env = suite.load(**env_config[c.ENV_BASE])
    elif env_config[c.ENV_TYPE] == c.GYM_THING:
        import gym
        from gym_thing.gym_thing import reacher_env, pusher_env, visual_pusher_env, visual_reacher_env
        env = gym.make(**env_config[c.ENV_BASE])
    elif env_config[c.ENV_TYPE] == c.MANIPULATOR_LEARNING:
        import manipulator_learning.sim.envs as manlearn_envs
        if c.KWARGS in env_config:
            env = getattr(manlearn_envs, env_config[c.ENV_BASE][c.ENV_NAME])(
                dense_reward=False, n_substeps=5, **env_config[c.KWARGS])
        else:
            env = getattr(manlearn_envs, env_config[c.ENV_BASE][c.ENV_NAME])(
                dense_reward=False, n_substeps=5)
    else:
        raise NotImplementedError

    for wrapper_config in env_config[c.ENV_WRAPPERS]:
        env = wrapper_config[c.WRAPPER](env, **wrapper_config[c.KWARGS])

    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)

    env.seed(seed)

    return env
