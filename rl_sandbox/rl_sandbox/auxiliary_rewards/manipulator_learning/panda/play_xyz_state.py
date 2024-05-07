import numpy as np
import torch
from numpy.linalg import norm
from rl_sandbox.auxiliary_rewards.manipulator_learning.panda.lift_xyz_state import close #, open_action, close_action


require_reach_radius = .1  # could be None -- for bring tasks, only get reward if reached in this radius,
                           # for other tasks, give a general sparse reward for being in this radius
include_pos_limits_penalty = True
all_rew_include_dense_reach = False
all_rew_reach_multiplier = .1
block_inds = [slice(0, 3), slice(7, 10)]  # block 0 inds, block 1 inds from info['obj_pos']
block_vel_inds = [slice(0, 3), slice(6, 9)]  # block 0 inds, block 1 inds from info['obj_vel']
num_blocks = len(block_inds)

obj_pos_slice = slice(12, 26)
grip_pos_slice = slice(6, 8)
pos_slice = slice(0, 3)
obj_vel_slice = slice(26, 38)
pick_and_place_obj_vel_slice = slice(29, 41)
pick_and_place_target_slice = slice(26, 29)  # uses pick and place env indices
tray_block_height = .145
stack_height = .185

include_unstack_in_aux = True

# for grip pos, 1 is fully open, 0 is fully closed, .52-.53 is closed on block


def dense_reach_bonus(task_rew, b_pos, arm_pos, max_reach_bonus=1.5, reach_thresh=.02,
                      reach_multiplier=all_rew_reach_multiplier):
    """ Convenience function for adding a conditional dense reach bonus to an aux task.

    If the task_rew is > 1, this indicates that the actual task is complete, and instead of giving a reach
    bonus, the max amount of reward given for a reach should be given (regardless of whether reach is satisfied).
    If it is < 1, a dense reach reward is given, and the actual task reward is given ONLY if the reach
    condition is satisfied. """
    if task_rew > 1:
        total_rew = task_rew + reach_multiplier * max_reach_bonus
    else:
        reach_rew = close(reach_thresh, b_pos, arm_pos, close_rew=max_reach_bonus)
        new_task_rew = task_rew * int(reach_rew > 1)
        total_rew = reach_multiplier * reach_rew + new_task_rew
    return total_rew


def pos_limits_penalty(e_info, action, penalty_mag=-.1):
# def pos_limits_penalty(e_info, action, penalty_mag=-.1):
    lim = e_info['at_limits'].flatten()
    penalty_quantities = penalty_mag * np.clip(np.concatenate([-action[:3], action[:3]]), 0, 1)
    penalties = lim * penalty_quantities

    return np.sum(penalties)
    # if np.any(lim[np.array([0, 1, 2, 3, 4, 5])]):
    #     return penalty_mag
    # else:
    #     return 0


def torch_multi_close_or_open(open_bool, acts, obss):
    action_mag = torch.norm(acts[:, :3], dim=-1)
    if open_bool:
        open_or_close_rew = (acts[:, -1] < 0).float()
    else:
        open_or_close_rew = (acts[:, -1] > 0).float()
    total_rew = open_or_close_rew - .5 * action_mag
    return total_rew


# OPEN AND CLOSE AUX
def close_open_gen(open_bool, include_reach, obs_act_only=False):
    """ If open_bool is False, then this is a close reward instead. """
    def close_or_open_action(info, action, observation, torch_multi=False, **kwargs):
        if torch_multi:
            assert obs_act_only, "obs_act_only must be True for torch_multi"
            assert not include_reach, "include_reach must be False for torch_multi"
            return torch_multi_close_or_open(open_bool, action, observation)

        observation = observation.squeeze()
        action_mag = norm(action[:3])
        if open_bool:
            open_or_close_rew = 1 if action[-1] < 0 else 0
        else:
            open_or_close_rew = 1 if action[-1] > 0 else 0
        total_rew = open_or_close_rew - .5 * action_mag

        if include_reach and require_reach_radius is not None:
            b0_pos = observation[obj_pos_slice][block_inds[0]]
            b1_pos = observation[obj_pos_slice][block_inds[1]]
            arm_pos = observation[pos_slice]
            reach_dist = min(norm(b0_pos - arm_pos), norm(b1_pos - arm_pos))
            total_rew *= int(reach_dist <= require_reach_radius)

        if include_pos_limits_penalty and not obs_act_only:
            e_info = info["infos"][-1]
            total_rew += pos_limits_penalty(e_info, action)

        return total_rew

    if open_bool:
        close_or_open_action.__qualname__ = "open_action" if include_reach else "pure_open"
    else:
        close_or_open_action.__qualname__ = "close_action" if include_reach else "pure_close"
    return close_or_open_action

open_action = close_open_gen(True, True)
close_action = close_open_gen(False, True)
pure_open = close_open_gen(True, False, True)
pure_close = close_open_gen(False, False, True)


# UNSTACK AUX
def unstack_gen(block=0, include_reach_bonus=all_rew_include_dense_reach, obs_act_only=True):
    """ Unstack block (1-block) from being on top of block."""
    block = 1 - block  # since we are interested in the height of opposite from main block

    def unstack_at(info, action, observation, **kwargs):
        if obs_act_only:
            observation = observation.squeeze()
            e_info = dict()
            e_info['obj_pos'] = observation[obj_pos_slice]
            e_info['grip_pos'] = observation[grip_pos_slice]
            e_info['pos'] = observation[pos_slice]
        else:
            e_info = info["infos"][-1]

        block_height = max(e_info['obj_pos'][block_inds[block]][2] - tray_block_height, 0)  # max in case block "sinks" thru tray
        b_pos = e_info['obj_pos'][block_inds[block]]
        arm_pos = e_info['pos']

        # reach_close = close(.01, b_pos, arm_pos)
        #
        # if close_shaping:
        #     grip_pos = np.array(e_info['grip_pos'])
        #
        #     close_bonus = .3 * int(reach_close > 1 and np.all(grip_pos < .7) and np.all(grip_pos > .4)
        #                            and (action[2] > .1 or block_height > .03))
        #     # close_bonus = .3 * int(reach_close > 1 and np.all(grip_pos < .7) and np.all(grip_pos > .4))
        #     # if reach_close > 1:
        #     #     close_bonus = .1 * int(np.all(grip_pos < .7) and np.all(grip_pos > .4) and block_height > .005)
        #     # else:
        #     #     close_bonus = .02 * int(reach_close < 1 and np.all(grip_pos > .7))
        # else:
        #     close_bonus = 0

        # unstack_rew = max((stack_height - tray_block_height - block_height) / .04, 0)  # max so that if lifted, doesn't go negative

        # if too low, means block sank below tray, if too high, means it rode up edges
        unstack_rew = int(tray_block_height - .001 <= e_info['obj_pos'][block_inds[block]][2] <= tray_block_height + .001)

        if require_reach_radius is not None:
            reach_dist = np.linalg.norm(arm_pos - b_pos)
            unstack_rew += all_rew_reach_multiplier * int(reach_dist <= require_reach_radius)

        if include_reach_bonus:
            unstack_rew = dense_reach_bonus(unstack_rew, b_pos, arm_pos, reach_thresh=.03)

        if include_pos_limits_penalty and not obs_act_only:
            unstack_rew += pos_limits_penalty(e_info, action)

        return unstack_rew

    unstack_at.__qualname__ = "unstack_" + str(1 - block)  # name MUST be unstack_X to be pickleable
    return unstack_at

unstack_0 = unstack_gen(0)
unstack_1 = unstack_gen(1)

# BRING AUX
def bring_gen(block=0, include_reach_bonus=all_rew_include_dense_reach, bring_pos_name='bring_goal_poss',
              bring_close_thresh=.02, remove_other_bonus_mult=None, obs_act_only=True, include_brought_bonus=True):
    """ Dense reward for reach, except when obj in goal pos.
        Dense reward for obj near goal pos, but only when reach within threshold OR when obj within threshold of goal.

        With required reach radius, instead, only gets reward if arm is somewhat reaching object."""
    def bring_at(info, action, observation, **kwargs):
        if obs_act_only:
            observation = observation.squeeze()
            e_info = dict()
            e_info['obj_pos'] = observation[obj_pos_slice]
            e_info['grip_pos'] = observation[grip_pos_slice]
            e_info['pos'] = observation[pos_slice]
            e_info['bring_goal_poss'] = np.array([[.075, .5, .145], [-.075, .5, .145]])
            e_info['insert_goal_poss'] = np.array([[.075, .5, .135], [-.075, .5, .135]])
        else:
            e_info = info["infos"][-1]
        e_info['pick_and_place_goal_poss'] = np.array(
            [observation[pick_and_place_target_slice]])  # will break on object 1
        b_pos = e_info['obj_pos'][block_inds[block]]
        other_b_pos = e_info['obj_pos'][block_inds[1 - block]]
        bring_pos = e_info[bring_pos_name][block]
        arm_pos = e_info['pos']

        bring_rew = close(bring_close_thresh, b_pos, bring_pos)

        arm_block_not_close_dense = np.clip(np.linalg.norm(arm_pos - b_pos) / .04, 0, 1)
        if include_brought_bonus:
            is_brought_bonus = (bring_rew > 1) * (action[-1] < 0) * arm_block_not_close_dense * .3
        else:
            is_brought_bonus = 0

        reach_dist = np.linalg.norm(arm_pos - b_pos)
        other_reach_dist = np.linalg.norm(arm_pos - other_b_pos)

        if require_reach_radius is not None:
            if remove_other_bonus_mult is None:
                bring_rew *= int(reach_dist <= require_reach_radius)
            else:
                bring_rew *= int(reach_dist <= require_reach_radius or other_reach_dist <= require_reach_radius)

        if include_reach_bonus:
            bring_rew = dense_reach_bonus(bring_rew, b_pos, arm_pos)

        if include_pos_limits_penalty and not obs_act_only:
            bring_rew += pos_limits_penalty(e_info, action)

        remove_other_bonus = 0
        if remove_other_bonus_mult is not None:
            remove_other_bonus = remove_other_bonus_mult * (1 - close(0, bring_pos, other_b_pos))
            if require_reach_radius is not None:
                remove_other_bonus *= int(reach_dist <= require_reach_radius or
                                          other_reach_dist <= require_reach_radius)

        return bring_rew + is_brought_bonus + remove_other_bonus
    bring_at.__qualname__ = "bring_" + str(block)  # name MUST be bring_X to be pickleable
    return bring_at

bring_0 = bring_gen(0)
bring_1 = bring_gen(1)


def insert_gen(block=0, include_reach_bonus=all_rew_include_dense_reach):
    insert_at = bring_gen(block, include_reach_bonus, 'insert_goal_poss', .001)
    insert_at.__qualname__ = "insert_" + str(block)   # name MUST be insert_X to be pickleable
    return insert_at

insert_0 = insert_gen(0)
insert_1 = insert_gen(1)


# PICK AND PLACE AUX
def pick_and_place_gen(block=0, include_reach_bonus=all_rew_include_dense_reach):
    pick_and_place_at = bring_gen(block, include_reach_bonus, 'pick_and_place_goal_poss', .008,
                                  include_brought_bonus=False)  # a little tighter than success
    pick_and_place_at.__qualname__ = "pick_and_place_" + str(block)  # name MUST be pick_and_place_X to be pickleable
    return pick_and_place_at

pick_and_place_0 = pick_and_place_gen(0)


# LIFT AUX
def lift_gen(block=0, max_rew_height=.08, block_on_table_height=.145, close_shaping=True,
             include_reach_bonus=all_rew_include_dense_reach):
    def lift(info, action, **kwargs):
        e_info = info["infos"][-1]
        block_height = e_info['obj_pos'][block_inds[block]][2] - block_on_table_height
        b_pos = e_info['obj_pos'][block_inds[block]]
        arm_pos = e_info['pos']

        reach_close = close(.01, b_pos, arm_pos)

        if close_shaping:
            grip_pos = np.array(e_info['grip_pos'])

            close_bonus = .3 * int(reach_close > 1 and np.all(grip_pos < .7) and np.all(grip_pos > .4)
                                   and (action[2] > .1 or block_height > .03))
            # close_bonus = .3 * int(reach_close > 1 and np.all(grip_pos < .7) and np.all(grip_pos > .4))
            # if reach_close > 1:
            #     close_bonus = .1 * int(np.all(grip_pos < .7) and np.all(grip_pos > .4) and block_height > .005)
            # else:
            #     close_bonus = .02 * int(reach_close < 1 and np.all(grip_pos > .7))
        else:
            close_bonus = 0

        if np.all(grip_pos < .75) and np.all(grip_pos > .4):  # so block doesn't just push up walls
            if block_height > max_rew_height:
                lift_rew = 1.5
            elif block_height < .005:
                lift_rew = 0
            else:
                lift_rew = block_height / max_rew_height
        else:
            lift_rew = 0

        if require_reach_radius is not None:
            reach_dist = np.linalg.norm(arm_pos - b_pos)
            lift_rew += all_rew_reach_multiplier * int(reach_dist <= require_reach_radius)

        if include_reach_bonus:
            lift_rew = dense_reach_bonus(lift_rew, b_pos, arm_pos, reach_thresh=.03)

        if include_pos_limits_penalty:
            lift_rew += pos_limits_penalty(e_info, action)

        if include_unstack_in_aux:
            # if too low, means block sank below tray, if too high, means it rode up edges, 0 if on tray, otherwise -.1
            unstack_pen = -.1 * (1 - int(
                tray_block_height - .001 <= e_info['obj_pos'][block_inds[1 - block]][2] <= tray_block_height + .001))
        else:
            unstack_pen = 0

        return lift_rew + close_bonus + unstack_pen
    lift.__qualname__ = "lift_" + str(block)  # name MUST be lift_X to be pickleable
    return lift

lift_0 = lift_gen(0)
lift_1 = lift_gen(1)


# STACK AUX
# blocks are 4cm tall, so "goal" is having block 0 be 4cm above block 1
def stack_gen(block=0, block_on_table_height=.145, include_reach_bonus=all_rew_include_dense_reach,
              include_lift_bonus=True, req_lift_height=.035, close_shaping=True, obs_act_only=True,
              pick_and_place_env=False):
    # only works if block = 0 or block = 1
    def stack_at(info, action, observation, **kwargs):
        if obs_act_only:
            observation = observation.squeeze()
            e_info = dict()
            e_info['obj_pos'] = observation[obj_pos_slice]
            e_info['grip_pos'] = observation[grip_pos_slice]
            e_info['pos'] = observation[pos_slice]
            if pick_and_place_env:
                e_info['obj_vel'] = observation[pick_and_place_obj_vel_slice]
            else:
                e_info['obj_vel'] = observation[obj_vel_slice]
        else:
            e_info = info["infos"][-1]
        b_pos = e_info['obj_pos'][block_inds[block]]
        stack_above = e_info['obj_pos'][block_inds[1 - block]] + np.array([0, 0, .041])
        stack_close = close(0.005, b_pos, stack_above)
        block_height = e_info['obj_pos'][block_inds[block]][2] - block_on_table_height
        is_lifted = block_height > req_lift_height
        stack_close = int(is_lifted) * stack_close
        grip_pos = np.array(e_info['grip_pos'])
        block_vel = e_info['obj_vel'][block_inds[block]]
        block_vel_mag = np.linalg.norm(block_vel)

        stack_dist = np.linalg.norm(b_pos - stack_above)

        # see if "stacked" and if arm isn't close to give a dense "stacked" bonus
        arm_pos = e_info['pos']
        arm_block_not_close_dense = np.clip(np.linalg.norm(arm_pos - b_pos) / .04, 0, 1)
        # is_stacked_bonus = int(stack_close > 1) * arm_block_not_close_dense  # > 1 when within close threshold
        is_stacked_bonus = is_lifted * (block_vel_mag < .01) * arm_block_not_close_dense

        if include_reach_bonus:
            stack_close = dense_reach_bonus(stack_close, b_pos, arm_pos)

        close_bonus = 0
        open_bonus = 0
        if close_shaping:
            reach_close = close(.01, b_pos, arm_pos)
            close_bonus = .1 * int(reach_close > 1 and np.all(grip_pos < .7) and np.all(grip_pos > .4)
                                   and (action[2] > .1 or block_height > .03))

            # action[3] < 0 corresponds to open. gives positive only reward for eventually opening
            if stack_dist < .005:
                open_bonus = 1 * (-action[3] + 1) * int(block_height > .03)

        lift_bonus = 0
        if include_lift_bonus:
            if block_height > .005 and np.all(grip_pos < .7) and np.all(grip_pos > .4):
                lift_bonus = 2 * all_rew_reach_multiplier * np.clip(block_height / req_lift_height, 0, req_lift_height)
            else:
                lift_bonus = 0

        # print("stack: ", stack_close, "is_stacked: ", is_stacked_bonus, "lift: ", lift_bonus, "close: ", close_bonus,
        #       "open: ", open_bonus)

        total_rew = stack_close + is_stacked_bonus + lift_bonus + close_bonus + open_bonus

        if require_reach_radius is not None:
            reach_dist = np.linalg.norm(arm_pos - b_pos)
            # total_rew *= all_rew_reach_multiplier * int(reach_dist <= require_reach_radius)
            total_rew *= int(reach_dist <= require_reach_radius)

        if include_pos_limits_penalty and not obs_act_only:
            total_rew += pos_limits_penalty(e_info, action)

        # add a penalty for block to be stacked on not being on tray
        stack_block_height = max(e_info['obj_pos'][block_inds[1 - block]][2] - tray_block_height, 0)  # max in case block "sinks" thru tray
        # unstack_rew = .1 * min(-stack_block_height / .04, 0)  # 0 if on tray, otherwise -.1

        # if too low, means block sank below tray, if too high, means it rode up edges, 0 if on tray, otherwise -.1
        if include_unstack_in_aux:
            unstack_pen = -.1 * (1 - int(
                tray_block_height - .001 <= e_info['obj_pos'][block_inds[1 - block]][2] <= tray_block_height + .001))
        else:
            unstack_pen = 0

        return total_rew + unstack_pen

    if pick_and_place_env:
        stack_at.__qualname__ = "stack_pp_env_" + str(block)  # name MUST be stack_X to be pickleable
    else:
        stack_at.__qualname__ = "stack_" + str(block)  # name MUST be stack_X to be pickleable
    return stack_at

stack_0 = stack_gen(0)
stack_1 = stack_gen(1)
stack_pp_env_0 = stack_gen(0, pick_and_place_env=True)
stack_pp_env_1 = stack_gen(1, pick_and_place_env=True)


# REACH AUX
def reach_gen(block=0):
    def reach(info, action, **kwargs):
        e_info = info["infos"][-1]
        close_rew = close(0.0, e_info['obj_pos'][block_inds[block]], e_info['pos'])

        if include_pos_limits_penalty:
            close_rew += pos_limits_penalty(e_info, action)

        # if too low, means block sank below tray, if too high, means it rode up edges, 0 if on tray, otherwise -.1
        if include_unstack_in_aux:
            unstack_pen = -.1 * (1 - int(
                tray_block_height - .001 <= e_info['obj_pos'][block_inds[1 - block]][
                    2] <= tray_block_height + .001))
        else:
            unstack_pen = 0

        return close_rew + unstack_pen
    reach.__qualname__ = "reach_" + str(block)  # name MUST be reach_X to be pickleable
    return reach

reach_0 = reach_gen(0)
reach_1 = reach_gen(1)


# TOGETHER AUX
def blocks_together(info, action, include_reach_bonus=all_rew_include_dense_reach, **kwargs):
    """ Same as bring: only gives reward if either one block is reached, or if both blocks are together """
    e_info = info["infos"][-1]
    b0_pos = e_info['obj_pos'][block_inds[0]]
    b1_pos = e_info['obj_pos'][block_inds[1]]
    arm_pos = e_info['pos']
    blocks_close_rew = close(.042, b0_pos, b1_pos)

    if require_reach_radius is not None:
        reach0_dist = np.linalg.norm(arm_pos - b0_pos)
        # reach1_dist = np.linalg.norm(arm_pos - b1_pos)
        blocks_close_rew *= int(reach0_dist <= require_reach_radius)

    if include_reach_bonus:
        block_arm_close_max_rew = 1.5
        if blocks_close_rew > 1:  # then give the same amount as reach bonus, regardless of whether reached or not
            total_rew = blocks_close_rew + all_rew_reach_multiplier * block_arm_close_max_rew
        else:
            b0_arm_close = close(.02, arm_pos, b0_pos, close_rew=block_arm_close_max_rew)
            b1_arm_close = close(.02, arm_pos, b1_pos, close_rew=block_arm_close_max_rew)
            reach_bonus = max(b0_arm_close, b1_arm_close)
            blocks_close_rew *= int(reach_bonus > 1)
            total_rew = blocks_close_rew + all_rew_reach_multiplier * reach_bonus
    else:
        total_rew = blocks_close_rew

    if include_pos_limits_penalty:
        total_rew += pos_limits_penalty(e_info, action)

    return total_rew


# MOVE AUX
def move_obj_gen(block=0, require_reach=True, include_lift_bonus=True, block_on_table_height=.145,
                 include_reach_bonus=all_rew_include_dense_reach, acc_pen_mult=.1):
    def move_obj(info, action, **kwargs):
        e_info = info["infos"][-1]
        obj_t_vel_mag = 5 * np.linalg.norm(e_info['obj_vel'][block_vel_inds[block]])  # since .3 is max, scale by 5
        b_pos = e_info['obj_pos'][block_inds[block]]
        arm_pos = e_info['pos']
        # is_close = np.linalg.norm(arm_pos - b_pos) < .02
        is_close = np.linalg.norm(arm_pos - b_pos) < .04

        if include_lift_bonus:
            block_height = e_info['obj_pos'][block_inds[block]][2] - block_on_table_height
            grip_pos = e_info['grip_pos']
            is_lifted = np.all(grip_pos < .7) and np.all(grip_pos > .4) and block_height > .005
            bonus = all_rew_reach_multiplier * int(is_lifted)
        else:
            bonus = 0

        if require_reach:  # without this could learn to pick up and drop
            obj_t_vel_mag *= int(is_close)

        if require_reach_radius is not None:
            reach_dist = np.linalg.norm(arm_pos - b_pos)
            obj_t_vel_mag += all_rew_reach_multiplier * int(reach_dist <= require_reach_radius)

        if include_reach_bonus:
            bonus += all_rew_reach_multiplier * close(.03, b_pos, arm_pos, close_rew=1.5)

        if include_pos_limits_penalty:
            obj_t_vel_mag += pos_limits_penalty(e_info, action)

        acc_pen = 0
        if acc_pen_mult is not None:
            obj_t_acc_mag = min(np.linalg.norm(e_info['obj_acc'][block_vel_inds[block]]), 1.5)
            obj_t_acc_mag *= int(is_close)
            acc_pen = acc_pen_mult * obj_t_acc_mag

        # if too low, means block sank below tray, if too high, means it rode up edges, 0 if on tray, otherwise -.1
        if include_unstack_in_aux:
            unstack_pen = -.1 * (1 - int(
                tray_block_height - .001 <= e_info['obj_pos'][block_inds[1 - block]][
                    2] <= tray_block_height + .001))
        else:
            unstack_pen = 0

        return obj_t_vel_mag + bonus - acc_pen + unstack_pen
    move_obj.__qualname__ = "move_obj_" + str(block)  # name MUST be move_obj_X to be pickleable
    return move_obj

move_obj_0 = move_obj_gen(0)
move_obj_1 = move_obj_gen(1)


# CLASSES
class AuxiliaryReward:
    def __init__(self, aux_rewards=(), include_main=True):
        self._aux_rewards = aux_rewards
        self._include_main = include_main

        # get a list of the aux rewards as strings
        ar_strs = []
        for ar in self._aux_rewards:
            ar_strs.append(ar.__qualname__)
        self._aux_rewards_str = ar_strs

    @property
    def num_auxiliary_rewards(self):
        return len(self._aux_rewards) + self._include_main

    def set_aux_rewards_str(self):
        """ For older loaded classes that don't call init. """
        ar_strs = []
        for ar in self._aux_rewards:
            ar_strs.append(ar.__qualname__)
        self._aux_rewards_str = ar_strs

    def reward(self, observation, action, reward, done, next_observation, info):
        observation = observation.reshape(-1)
        next_observation = next_observation.reshape(-1)
        reward_vector = []
        if self._include_main:
            reward_vector.append(reward)
        for task_reward in self._aux_rewards:
            reward_vector.append(task_reward(
                observation=observation, action=action, reward=reward, next_observation=next_observation,
                done=done, info=info))
        return np.array(reward_vector, dtype=np.float32)


# some convenience functions for generating each play class
aux_rewards_all = [open_action, close_action]
# aux_rewards_all = []


def get_aux_rewards(block, aux_rewards_added_str):
    aux_rewards_added = []
    if block is None:
        blocks = list(range(num_blocks))
    else:
        blocks = [block]
    for i in blocks:
        aux_rewards_added.extend([globals()[f_str + '_' + str(i)] for f_str in aux_rewards_added_str])
    return aux_rewards_added


class PandaPlayXYZStateAuxiliaryReward(AuxiliaryReward):
    """ Play aux reward class set up to take main_task as argument. Should be of form:
        {main_task}_{block_index}, for tasks where a single block are either optional or mandatory.

        Examples:
            - all
            - stack_01  (stack must have 01 or 10)
            - insert  (both)
            - insert_0
            - lift_0 (lift must have 0 or 1)
    """
    def __init__(self, main_task, include_main=True, aux_rewards_all=aux_rewards_all):
        und_loc = main_task.rfind('_')
        if und_loc > -1:
            main_task_no_suf = main_task[:und_loc]
            block_suf = [int(suf_char) for suf_char in list(main_task[und_loc + 1:])]

            # hard-coded if argument is 01 or 10 for stack, only take 0 or 1
            block_suf = block_suf[0]
        else:
            main_task_no_suf = main_task
            block_suf = None

        if main_task_no_suf == 'all':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'insert', 'bring', 'lift', 'reach', 'move_obj'])
            aux_rewards_added.append(blocks_together)
        elif main_task_no_suf == 'stack':
            # print("Warning: Ensure that env is PandaPlayInsertTrayXYZState. If using "
            #       "PandaPlayInsertTrayPlusPickPlaceXYZState, main_task should be stack_pp_env_X.")
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'lift', 'reach', 'move_obj'])
            # aux_rewards_added.append(open_action)
        elif main_task_no_suf == 'stack_no_move':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'lift', 'reach'])
        elif main_task_no_suf == 'stack_pp_env':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack_pp_env', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'insert':
            aux_rewards_added = get_aux_rewards(block_suf, ['insert', 'bring', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'insert_no_bring':
            aux_rewards_added = get_aux_rewards(block_suf, ['insert', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'insert_no_bring_no_move':
            aux_rewards_added = get_aux_rewards(block_suf, ['insert', 'lift', 'reach'])
        elif main_task_no_suf == 'bring':
            aux_rewards_added = get_aux_rewards(block_suf, ['bring', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'bring_no_move':
            aux_rewards_added = get_aux_rewards(block_suf, ['bring', 'lift', 'reach'])
        elif main_task_no_suf == 'lift':
            # aux_rewards_added = get_aux_rewards(block_suf, ['lift', 'reach', 'move_obj'])
            aux_rewards_added = get_aux_rewards(block_suf, ['lift', 'reach'])
        elif main_task_no_suf == 'move_obj':
            aux_rewards_added = get_aux_rewards(block_suf, ['lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'reach':
            aux_rewards_all = [open_action]  # remove close_action for reach
            aux_rewards_added = get_aux_rewards(block_suf, ['reach'])
        elif main_task_no_suf == 'together':
            aux_rewards_added = get_aux_rewards(block_suf, ['lift', 'reach', 'move_obj'])
            aux_rewards_added.append(blocks_together)
        elif main_task_no_suf == 'bring_and_remove':
            aux_rewards_added = get_aux_rewards(block_suf, ['bring', 'lift', 'reach', 'move_obj'])
            aux_rewards_added += get_aux_rewards(1 - block_suf, ['lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'lift_open_close':
            aux_rewards_added = get_aux_rewards(block_suf, ['lift'])
        elif main_task_no_suf == 'stack_open_close':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack'])
        elif main_task_no_suf == 'pick_and_place':
            aux_rewards_added = get_aux_rewards(block_suf, ['pick_and_place', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'unstack_stack':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'unstack', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'unstack_stack_env_only':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'unstack_stack_env_only_no_move':
            aux_rewards_added = get_aux_rewards(block_suf, ['stack', 'lift', 'reach'])
        elif main_task_no_suf == 'unstack_move_obj':
            aux_rewards_added = get_aux_rewards(block_suf, ['unstack', 'lift', 'reach', 'move_obj'])
        elif main_task_no_suf == 'unstack_lift':
            aux_rewards_added = get_aux_rewards(block_suf, ['unstack', 'lift', 'reach'])
        else:
            raise NotImplementedError("PandaPlayXYZStateAuxiliaryReward not implemented for main_task %s" % main_task)
        # if we want to make the main task the first task, switch this comment
        # super().__init__(aux_rewards=tuple(aux_rewards_added + aux_rewards_all), include_main=include_main)
        super().__init__(aux_rewards=tuple(aux_rewards_all + aux_rewards_added), include_main=include_main)