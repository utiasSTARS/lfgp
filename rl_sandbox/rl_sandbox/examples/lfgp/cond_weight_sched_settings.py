import numpy as np


def get_cond_scheduler_settings(aux_reward_obj):
    rew_names = [f.__qualname__ for f in aux_reward_obj._aux_rewards]

    # for stack
    if set(rew_names) == {'open_action', 'close_action', 'stack_0', 'lift_0', 'reach_0', 'move_obj_0'}:
        t = {'o': 'open_action', 'c': 'close_action', 's': 'stack_0', 'l': 'lift_0', 'r': 'reach_0', 'm': 'move_obj_0'}
        task_reset_probs_dict = {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5}
        task_conditional_probs_dict = {
            t['o']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
            t['c']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
            t['s']: {t['o']: 0.4, t['c']: 0.0, t['s']: 0.4, t['l']: 1/3*.2, t['r']: 1/3*.2, t['m']: 1/3*.2},
            t['l']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['l']: 0.25, t['r']: 0.0, t['m']: 0.25},
            t['r']: {t['o']: 0.0, t['c']: .125, t['s']: 0.5, t['l']: 0.125, t['r']: 0.125, t['m']: 0.125},
            t['m']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
        }

    # for unstack move obj
    # elif set(rew_names) == {'open_action', 'close_action', 'unstack_0', 'lift_0', 'reach_0', 'move_obj_0'}:
    #     t = {'o': 'open_action', 'c': 'close_action', 'u': 'unstack_0', 'l': 'lift_0', 'r': 'reach_0', 'm': 'move_obj_0'}
    #     task_reset_probs_dict = {t['o']: 1/4*.5, t['c']: 0.0, t['u']: 0.5, t['l']: 1/4*.5, t['r']: 1/4*.5, t['m']: 1/4*.5}
    #     task_conditional_probs_dict = {
    #         t['o']: {t['o']: 0.0, t['c']: 0.0, t['u']: 1/3*0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: .5},
    #         t['c']: {t['o']: 0.0, t['c']: 0.0, t['u']: 1/3*0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: .5},
    #         t['u']: {t['o']: 0.1, t['c']: 0.0, t['u']: 0.2, t['l']: .1, t['r']: .1, t['m']: .5},
    #         t['l']: {t['o']: 0.0, t['c']: 0.0, t['u']: 0.2, t['l']: 0.4, t['r']: 0.0, t['m']: 0.4},
    #         t['r']: {t['o']: 0.0, t['c']: .125, t['u']: 0.125, t['l']: 0.125, t['r']: 0.125, t['m']: 0.5},
    #         t['m']: {t['o']: 1/3*.4, t['c']: 0.0, t['u']: 0.0, t['l']: 1/3*.4, t['r']: 1/3*.4, t['m']: .6},
    #     }
    
    # for unstack move obj, but encourages lift --> move --> lift more
    elif set(rew_names) == {'open_action', 'close_action', 'unstack_0', 'lift_0', 'reach_0', 'move_obj_0'}:
        t = {'o': 'open_action', 'c': 'close_action', 'u': 'unstack_0', 'l': 'lift_0', 'r': 'reach_0', 'm': 'move_obj_0'}
        task_reset_probs_dict = {t['o']: 1/4*.5, t['c']: 0.0, t['u']: 0.5, t['l']: 1/4*.5, t['r']: 1/4*.5, t['m']: 1/4*.5}
        task_conditional_probs_dict = {
            t['o']: {t['o']: 0.0, t['c']: 0.0, t['u']: 1/3*0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: .5},
            t['c']: {t['o']: 0.0, t['c']: 0.0, t['u']: 1/3*0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: .5},
            t['u']: {t['o']: 0.1, t['c']: 0.0, t['u']: 0.2, t['l']: .1, t['r']: .1, t['m']: .5},
            t['l']: {t['o']: 0.0, t['c']: 0.0, t['u']: 0.2, t['l']: 0.2, t['r']: 0.0, t['m']: 0.6},
            t['r']: {t['o']: 0.0, t['c']: .125, t['u']: 0.125, t['l']: 0.125, t['r']: 0.125, t['m']: 0.5},
            t['m']: {t['o']: .1, t['c']: 0.0, t['u']: 0.0, t['l']: .6, t['r']: .1, t['m']: .2},
        }
    
    # for unstack stack

    # 08-25-21_09_44_20
    # elif set(rew_names) == {'open_action', 'close_action', 'stack_0', 'unstack_0', 'lift_0', 'reach_0', 'move_obj_0'}:
    #     t = {'o': 'open_action', 'c': 'close_action', 's': 'stack_0', 'u': 'unstack_0', 'l': 'lift_0', 'r': 'reach_0', 'm': 'move_obj_0'}
    #     task_reset_probs_dict = {t['o']: 0.0, t['c']: 0.0, t['s']: 0.2, t['u']: 0.3, t['l']: 1/3*.5, t['r']: 1/3*.5, 
    #                              t['m']: 1/3*.5}
    #     task_conditional_probs_dict = {
    #         t['o']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.0, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
    #         t['c']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.0, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
    #         t['s']: {t['o']: 0.4, t['c']: 0.0, t['s']: 0.4, t['u']: 0.0, t['l']: 1/3*.2, t['r']: 1/3*.2, t['m']: 1/3*.2},
    #         t['u']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.125, t['l']: .125, t['r']: .125, t['m']: .125},
    #         t['l']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.0, t['l']: 0.25, t['r']: 0.0, t['m']: 0.25},
    #         t['r']: {t['o']: 0.0, t['c']: .125, t['s']: 0.5, t['u']: 0.0, t['l']: 0.125, t['r']: 0.125, t['m']: 0.125},
    #         t['m']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.0, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
    #     }
    

    elif set(rew_names) == {'open_action', 'close_action', 'stack_0', 'unstack_0', 'lift_0', 'reach_0', 'move_obj_0'}:
        t = {'o': 'open_action', 'c': 'close_action', 's': 'stack_0', 'u': 'unstack_0', 'l': 'lift_0', 'r': 'reach_0', 'm': 'move_obj_0'}
        task_reset_probs_dict = {t['o']: 0.0, t['c']: 0.0, t['s']: 0.35, t['u']: 0.35, t['l']: .1, t['r']: .1, 
                                 t['m']: .1}
        task_conditional_probs_dict = {
            t['o']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.0, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
            t['c']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.0, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
            t['s']: {t['o']: 0.4, t['c']: 0.0, t['s']: 0.4, t['u']: 0.0, t['l']: 1/3*.2, t['r']: 1/3*.2, t['m']: 1/3*.2},
            t['u']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.2, t['l']: .1, t['r']: .1, t['m']: .1},
            t['l']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.0, t['l']: 0.25, t['r']: 0.0, t['m']: 0.25},
            t['r']: {t['o']: 0.0, t['c']: .125, t['s']: 0.5, t['u']: 0.0, t['l']: 0.125, t['r']: 0.125, t['m']: 0.125},
            t['m']: {t['o']: 0.0, t['c']: 0.0, t['s']: 0.5, t['u']: 0.0, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
        }

    # for bring
    elif set(rew_names) == {'open_action', 'close_action', 'bring_0', 'lift_0', 'reach_0', 'move_obj_0'}:
        t = {'o': 'open_action', 'c': 'close_action', 'b': 'bring_0', 'l': 'lift_0', 'r': 'reach_0', 'm': 'move_obj_0'}
        task_reset_probs_dict = {t['o']: 0.0, t['c']: 0.0, t['b']: 0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5}
        task_conditional_probs_dict = {
            t['o']: {t['o']: 0.0, t['c']: 0.0, t['b']: 0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
            t['c']: {t['o']: 0.0, t['c']: 0.0, t['b']: 0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
            t['b']: {t['o']: 0.4, t['c']: 0.0, t['b']: 0.4, t['l']: 1/3*.2, t['r']: 1/3*.2, t['m']: 1/3*.2},
            t['l']: {t['o']: 0.0, t['c']: 0.0, t['b']: 0.5, t['l']: 0.25, t['r']: 0.0, t['m']: 0.25},
            t['r']: {t['o']: 0.0, t['c']: .125, t['b']: 0.5, t['l']: 0.125, t['r']: 0.125, t['m']: 0.125},
            t['m']: {t['o']: 0.0, t['c']: 0.0, t['b']: 0.5, t['l']: 1/3*.5, t['r']: 1/3*.5, t['m']: 1/3*.5},
        }

    # for insert
    elif set(rew_names) == {'open_action', 'close_action', 'insert_0', 'bring_0', 'lift_0', 'reach_0', 'move_obj_0'}:
        t = {'o': 'open_action', 'c': 'close_action', 'i': 'insert_0', 'b': 'bring_0', 'l': 'lift_0', 'r': 'reach_0', 'm': 'move_obj_0'}
        task_reset_probs_dict = {t['o']: 0.0, t['c']: 0.0, t['i']: 0.5, t['b']: .125, t['l']: .125, t['r']: .125,
                                 t['m']: .125}

        task_conditional_probs_dict = {
            t['o']: {t['o']: 0.0, t['c']: 0.0, t['i']: 0.5, t['b']: 0.125, t['l']: .125, t['r']: .125, t['m']: .125},
            t['c']: {t['o']: 0.0, t['c']: 0.0, t['i']: 0.5, t['b']: 0.125, t['l']: .125, t['r']: .125, t['m']: .125},
            t['i']: {t['o']: 0.4, t['c']: 0.0, t['i']: 0.4, t['b']: 0.05, t['l']: .05, t['r']: .05, t['m']: .05},
            t['b']: {t['o']: 0.0, t['c']: 0.0, t['i']: 0.8, t['b']: 0.05, t['l']: .05, t['r']: .05, t['m']: .05},
            t['l']: {t['o']: 0.0, t['c']: 0.0, t['i']: 0.5, t['b']: 1/3*.5, t['l']: 1/3*.5, t['r']: 0, t['m']: 1/3*.5},
            t['r']: {t['o']: 0.0, t['c']: 0.1, t['i']: 0.5, t['b']: 0.1, t['l']: .1, t['r']: .1, t['m']: .1},
            t['m']: {t['o']: 0.0, t['c']: 0.0, t['i']: 0.5, t['b']: 0.125, t['l']: .125, t['r']: .125, t['m']: .125},
        }
    
    # for pick and place
    elif set(rew_names) == {'open_action', 'close_action', 'pick_and_place_0', 'lift_0', 'reach_0', 'move_obj_0'}:
        t = {'o': 'open_action', 'c': 'close_action', 'b': 'pick_and_place_0', 'l': 'lift_0', 'r': 'reach_0', 'm': 'move_obj_0'}
        task_reset_probs_dict = {t['o']: .1, t['c']: 0.0, t['b']: 0.5, t['l']: .4/3, t['r']: .4/3, t['m']: .4/3}
        task_conditional_probs_dict = {
            t['o']: {t['o']: 0.0, t['c']: 0.0, t['b']: 0.7, t['l']: .1, t['r']: .1, t['m']: .1},
            t['c']: {t['o']: 0.0, t['c']: 0.0, t['b']: 0.6, t['l']: .2, t['r']: 0.0, t['m']: .2},
            t['b']: {t['o']: 0.0, t['c']: 0.0, t['b']: 0.7, t['l']: 0.0, t['r']: 0.0, t['m']: .3},
            t['l']: {t['o']: 0.0, t['c']: 0.0, t['b']: 0.6, t['l']: 0.2, t['r']: 0.0, t['m']: 0.2},
            t['r']: {t['o']: 0.0, t['c']: .6, t['b']: 0.1, t['l']: 0.1, t['r']: 0.1, t['m']: 0.1},
            t['m']: {t['o']: 0.0, t['c']: 0.1, t['b']: 0.6, t['l']: 0.0, t['r']: 0.0, t['m']: 0.3},
        }
    else:
        raise NotImplementedError("No existing conditional scheduler settings for rewards named: %s" % rew_names)

    # convert to lists
    task_reset_probs = []
    task_conditional_probs = []
    for name in rew_names:
        task_reset_probs.append(task_reset_probs_dict[name])
        task_conditional_probs.append([])
        for sub_name in rew_names:
            task_conditional_probs[-1].append(task_conditional_probs_dict[name][sub_name])

    return np.array(task_reset_probs), np.array(task_conditional_probs)