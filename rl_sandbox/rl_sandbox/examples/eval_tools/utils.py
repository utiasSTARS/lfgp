import _pickle as pickle
import os
import torch

import rl_sandbox.constants as c

from rl_sandbox.agents.hrl_agents import SACXAgent, SACXPlusHandcraftAgent
from rl_sandbox.agents.rl_agents import ACAgent
from rl_sandbox.algorithms.sac_x.schedulers import FixedScheduler, WeightedRandomScheduler, WeightedRandomSchedulerPlusHandcraft
from rl_sandbox.envs.utils import make_env
from rl_sandbox.model_architectures.utils import make_model
from rl_sandbox.train.train_lfgp_sac import train_lfgp_sac


def load_model(seed, config_path, model_path, intention=0, device="cpu", include_env=True, include_disc=True,
               load_whole_agent=True, model_timestep=0, force_egl=False):
    assert os.path.isfile(model_path)
    assert os.path.isfile(config_path)

    discriminator = None

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    config[c.DEVICE] = torch.device(device)

    # config[c.ENV_SETTING][c.KWARGS]["egl"] = True
    buffer_preprocessing = config[c.BUFFER_PREPROCESSING]

    if config[c.ALGO] == c.MULTITASK_BC:
        model = make_model(config[c.MODEL_SETTING])
        scheduler = FixedScheduler(intention_i=intention, num_tasks=config[c.NUM_TASKS])
        model_data = torch.load(model_path, map_location=device)[c.STATE_DICT]
        model.load_state_dict(model_data)
        agent = SACXAgent(scheduler=scheduler,
                          intentions=model,
                          learning_algorithm=None,
                          scheduler_period=c.MAX_INT,
                          preprocess=config[c.EVALUATION_PREPROCESSING])

    elif config[c.ALGO] in (c.SACX, c.LFGP, c.LFGP_NS, "dacx"):
        if load_whole_agent and config[c.ALGO] in [c.LFGP, "dacx"]:
            config[c.LOAD_MODEL] = model_path

            if config[c.SCHEDULER_SETTING][c.TRAIN][c.MODEL_ARCHITECTURE] != WeightedRandomScheduler \
                    and config[c.SCHEDULER_SETTING][c.TRAIN][c.MODEL_ARCHITECTURE] != WeightedRandomSchedulerPlusHandcraft:
                # fix bug where scheduler temperature was not being saved
                skw = config[c.SCHEDULER_SETTING][c.TRAIN][c.KWARGS]
                num_decays = model_timestep / config[c.MAX_EPISODE_LENGTH] * skw[c.MAX_SCHEDULE]
                temperature = max(skw[c.TEMPERATURE_MIN], skw[c.TEMPERATURE] * skw[c.TEMPERATURE_DECAY] ** num_decays)
                config[c.SCHEDULER_SETTING][c.TRAIN][c.KWARGS][c.TEMPERATURE] = temperature

            # set all devices to new device
            config[c.INTENTIONS_SETTING][c.KWARGS][c.DEVICE] = device
            config[c.DISCRIMINATOR_SETTING][c.KWARGS][c.DEVICE] = device
            config[c.BUFFER_SETTING][c.KWARGS][c.DEVICE] = device

            agent = train_lfgp_sac(config, return_agent_only=True, no_expert_buffers=True)
            discriminator = agent.learning_algorithm.update_intentions.discriminator

        else:

            config[c.INTENTIONS_SETTING][c.KWARGS][c.DEVICE] = config[c.DEVICE]
            intentions = make_model(config[c.INTENTIONS_SETTING])
            intentions_model = torch.load(model_path, map_location=device)[c.INTENTIONS]
            if c.ALGORITHM in intentions_model.keys():
                intentions.load_state_dict(intentions_model[c.ALGORITHM][c.STATE_DICT])
            else:
                intentions.load_state_dict(intentions_model[c.STATE_DICT])

            scheduler = FixedScheduler(intention_i=intention,
                                       num_tasks=config[c.SCHEDULER_SETTING][c.TRAIN][c.KWARGS][c.NUM_TASKS])

            if config.get(c.HANDCRAFT_TASKS, None) is None:
                agent = SACXAgent(scheduler=scheduler,
                                  intentions=intentions,
                                  learning_algorithm=None,
                                  scheduler_period=c.MAX_INT,
                                  preprocess=config[c.EVALUATION_PREPROCESSING])
            else:
                agent = SACXPlusHandcraftAgent(scheduler=scheduler,
                                               intentions=intentions,
                                               learning_algorithm=None,
                                               scheduler_period=c.MAX_INT,
                                               preprocess=config[c.EVALUATION_PREPROCESSING])

            if config[c.ALGO] == c.LFGP:
                config[c.DISCRIMINATOR_SETTING][c.KWARGS][c.DEVICE] = config[c.DEVICE]
                discriminator = make_model(config[c.DISCRIMINATOR_SETTING])
                discriminator.load_state_dict(intentions_model[c.DISCRIMINATOR])

    else:
        config[c.MODEL_SETTING][c.KWARGS][c.DEVICE] = device
        model = make_model(config[c.MODEL_SETTING])

        saved_model = torch.load(model_path, map_location=device)
        if config[c.ALGO] == c.DAC:
            # also get discriminator
            config[c.DISCRIMINATOR_SETTING][c.KWARGS][c.DEVICE] = config[c.DEVICE]
            discriminator = make_model(config[c.DISCRIMINATOR_SETTING])
            discriminator.load_state_dict(saved_model[c.DISCRIMINATOR])

            saved_model = saved_model[c.ALGORITHM]

        model.load_state_dict(saved_model[c.STATE_DICT])

        if hasattr(model, c.OBS_RMS):
            model.obs_rms = saved_model[c.OBS_RMS]

        agent = ACAgent(model=model,
                        learning_algorithm=None,
                        preprocess=config[c.EVALUATION_PREPROCESSING])

    return_list = [config]

    if include_env:
        env_setting = config[c.ENV_SETTING]
        if force_egl:
            env_setting[c.KWARGS]["egl"] = True
        env = make_env(env_setting, seed=seed)
        return_list.append(env)

    return_list.extend([buffer_preprocessing, agent])
    if include_disc and discriminator is not None: return_list.append(discriminator)

    return tuple(return_list)
