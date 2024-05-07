import os
import rl_sandbox.constants as c

def get_save_path(algo_name, main_task, seed, exp_name, top_path="results"):
    return os.path.join(top_path, main_task, str(seed), algo_name, exp_name)


def config_check(experiment_config, top_path):
    """
    custom checks for fixing config for particular machines
    """
    if "scratch" in top_path and experiment_config[c.ENV_SETTING][c.ENV_TYPE] == c.MANIPULATOR_LEARNING:
        experiment_config[c.ENV_SETTING][c.KWARGS]["egl"] = False