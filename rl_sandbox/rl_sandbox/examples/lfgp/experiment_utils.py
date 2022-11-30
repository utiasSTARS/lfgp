import os
import rl_sandbox.constants as c

def get_save_path(exp_type, main_task, seed, exp_name, user_machine):
    """
    exp_type should be one of lfgp, bc, sacx, multitask_bc, dac, lfgp_ns
    """
    if user_machine == 'local':
        save_path = f"./results/{main_task}/{seed}/{exp_type}/{exp_name}"
    elif user_machine == "None":
        save_path = None
    elif user_machine == "starslab":
        main_save_path = f"/media/starslab/users/trevor-ablett/dac-x/play_xyz/"
        save_path = os.path.join(main_save_path, main_task, str(seed), exp_type, exp_name)
    elif user_machine == "trevor_cc":
        main_save_path = f"/home/abletttr/scratch/lfgp/"
        save_path = os.path.join(main_save_path, main_task, str(seed), exp_type, exp_name)
    else:
        raise NotImplementedError("Invalid option for argument user_machine of %s" % user_machine)

    return save_path


def config_check(experiment_config, user_machine):
    """
    custom checks for fixing config for particular user machines
    """
    if user_machine == 'trevor_cc':
        experiment_config[c.ENV_SETTING][c.KWARGS]["egl"] = False