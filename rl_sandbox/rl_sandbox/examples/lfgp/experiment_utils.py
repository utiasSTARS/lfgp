import os

def get_save_path(exp_type, main_task, seed, exp_name, user_machine):
    """
    exp_type should be one of lfgp, bc, sacx, multitask_bc, dac, lfgp_ns
    """
    if user_machine == 'local':
        save_path = f"./results/{main_task}/{seed}/{exp_type}/{exp_name}"
    elif user_machine == "None":
        save_path = None
    else:
        raise NotImplementedError("Invalid option for argument user_machine of %s" % user_machine)
    
    return save_path