import numpy as np
import tqdm
import copy
import importlib
import pickle

import rl_sandbox.envs.rce_envs as rce_envs
import d4rl.hand_manipulation_suite.raw_human_demonstrations as hand_dapg_demos


def do_sim_for_steps(env, num_steps=10, open_gripper=True):
    for _ in range(num_steps):
        if open_gripper:
            env.do_simulation([-1, 1], env.frame_skip)
        else:
            env.do_simulation([1, -1], env.frame_skip)

def set_and_sim_ee_pos(env, pos, num_steps=10, open_gripper=True):
    env.data.set_mocap_pos('mocap', pos)
    env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    do_sim_for_steps(env, num_steps=num_steps, open_gripper=open_gripper)


def env_render_test(env, grasp):
    print(f"before render: {env._get_obs()}")
    for _ in range(2):
        if grasp:
            env.step(np.array([0., 0., 0., 1.]))
        else:
            env.step(np.array([0., 0., 0., -.01]))
        env.render()
    print(f"after render: {env._get_obs()}")


def generic_reach(env, noise_mag=(0., 0., 0.), grasp=False):
    # get reach obs: reset-->sim-->set obj to (noisy) obj pos-->sim-->setting ee pos to (noisy) obj pos-->sim-->get obs
    test_render = False

    do_sim = True
    while do_sim:
        do_sim = False

        # reset-->sim
        env.reset()
        do_sim_for_steps(env)

        reset_obs = env._get_obs()
        reset_ee_pos = reset_obs[:3]
        reset_obj_pos = reset_obs[3:6]
        new_obj_pos = copy.deepcopy(reset_obj_pos)

        # add noise
        if noise_mag != (0., 0., 0.):
            nm = np.array(noise_mag) / 2
            noise = np.random.uniform(low=[-nm[0], -nm[1], 0], high=[nm[0], nm[1], nm[2] * 2])  # only positive in z
            new_obj_pos += noise

        # set obj to (noisy) obj pos-->sim
        if issubclass(type(env), rce_envs.SawyerDrawerOpen) or issubclass(type(env), rce_envs.SawyerDrawerClose):
            # at reset, the close one has env.data.qpos[9] = -.2, obj_pos = 0, 0.5, 0.09
            # at reset, the open one has env.data.qpos[9] = 0.0, obj_pos = 0, 0.7, 0.09
            # but set_obj_xyz sets qpos[9], so you must set accordingly based on actual obs
            env._set_obj_xyz(new_obj_pos[1] - .7)
            # ee_z_add = .07
            ee_z_add = .02
        elif issubclass(type(env), rce_envs.SawyerBoxClose):
            env._set_obj_xyz_quat(new_obj_pos, 0.0)
            ee_z_add = 0.0
        else:
            env._set_obj_xyz_quat(new_obj_pos, 0.0)
            ee_z_add = .02

        do_sim_for_steps(env)

        # set ee pos to (noisy) obj pos with top down reach-->sim-->obj pos-->sim
        ee_pos = copy.deepcopy(new_obj_pos)
        # ee_pos[2] += .02  # prevents end effector from hitting table and possibly bouncing
        ee_pos[2] += ee_z_add  # prevents end effector from hitting table and possibly bouncing

        # force a middle ee position to ensure no collision with drawer prior to reach
        first_ee_pos = copy.deepcopy(new_obj_pos)
        first_ee_pos[2] = reset_ee_pos[2]
        set_and_sim_ee_pos(env, first_ee_pos)
        set_and_sim_ee_pos(env, ee_pos)

        # add a grasp, always resetting object position so it doesn't fall while teeth close
        if grasp:
            # set_and_sim_ee_pos(env, ee_pos, num_steps=20, open_gripper=False)
            if issubclass(type(env), rce_envs.SawyerBinPicking) or \
                    issubclass(type(env), rce_envs.SawyerLift) or \
                    issubclass(type(env), rce_envs.SawyerPush):
                ee_pos[2] += .02

            if issubclass(type(env), rce_envs.SawyerBoxClose):
                new_obj_pos[2] -= .09

            for _ in range(20):
                set_and_sim_ee_pos(env, ee_pos, num_steps=1, open_gripper=False)
                if issubclass(type(env), rce_envs.SawyerDrawerOpen) or issubclass(type(env), rce_envs.SawyerDrawerClose):
                    env._set_obj_xyz(new_obj_pos[1] - .7)
                else:
                    env._set_obj_xyz_quat(new_obj_pos, 0.0)

        # y tolerance to potentially redo
        cur_obs = env._get_obs()
        if np.linalg.norm(cur_obs[1] - cur_obs[4]) > .02:
            do_sim = True

    if test_render:
        env_render_test(env, grasp)
        # import ipdb; ipdb.set_trace()

    return env._get_obs()


def sawyer_aux_reach_gen(base_env_class_inst, noise_mag=(0., 0., 0.), grasp=False):
    AuxReachClass = type('AuxReachClass', (type(base_env_class_inst),), {})

    def aux_init(self):
        super(AuxReachClass, self).__init__()

    def aux_get_expert_obs(self):
        return generic_reach(self, noise_mag=noise_mag, grasp=grasp)

    AuxReachClass.__init__ = aux_init
    AuxReachClass._get_expert_obs = aux_get_expert_obs

    return AuxReachClass


def hand_dapg_get_dataset(env, base_env, mode='reach', allow_multiple=False, vel_to_zero=True, terminal_offset=50,
                          max_num_demos=None, get_imgs=False, img_wh=(500, 500), start_demo=0):
    # mode options: 'grasp', 'reach', 'final'
    # actually step through the base environment and get the data on the way
    # first get the expert data
    test_render = False
    door_reach_threshold = .045
    # door_grasp_threshold = 0.03  # straight handle
    door_grasp_threshold = 1.4  # turned handle
    hammer_reach_threshold = 0.035
    hammer_grasp_threshold = 0.05  # starts at .0346 and then falls on to table at .0267
    relocate_reach_threshold = .045
    relocate_grasp_threshold = .05

    try:
        env_id = base_env.spec.id
    except:
        # handles case if we're using new derived envs with different obs
        class_str = type(base_env).__name__
        env_type_str = class_str.split('Env')[0].lower()
        env_id = f"{env_type_str}-human-v0"

    data_name = f"{env_id.split('-human-v0')[0]}-v0_demos.pickle"
    demos = pickle.load(importlib.resources.files(hand_dapg_demos).joinpath(data_name).open('rb'))

    expert_obs = []
    expert_imgs = []

    num_demos = len(demos) if max_num_demos is None else max_num_demos

    for d_i in tqdm.trange(start_demo, start_demo + num_demos):
        demo = demos[d_i]
        base_env.reset()
        base_env.set_env_state(demo['init_state_dict'])
        acts = demo['actions']
        reached = False
        grasped = False
        num_ep_exp = 0

        for t, act in enumerate(acts):
            obs, rew, done, info = base_env.step(act)

            if test_render:
                base_env.render()

            # add data based on thresholds, set thresholds based on individual envs
            if "door" in env_id:
                reach_dist = np.linalg.norm(obs[-4:-1])  # palm to handle dist
                grasp_dist = obs[27]  # latch position, >0 means starting to turn

                if d_i == 3:  # has an odd reach with different threshold
                    ep_reach_thresh = .063
                else:
                    ep_reach_thresh = door_reach_threshold

                ep_grasp_thresh = door_grasp_threshold

            elif "hammer" in env_id:
                reach_dist = np.linalg.norm(np.array(obs[-13:-10])-np.array(obs[-10:-7]))  # palm hammer dist
                grasp_dist = obs[-8]  # hammer height
                ep_reach_thresh = hammer_reach_threshold
                ep_grasp_thresh = hammer_grasp_threshold

                if vel_to_zero:
                    vel_indices = slice(27, 33)
                    obs[vel_indices] = 0.0

            elif "relocate" in env_id:
                reach_dist = np.linalg.norm(obs[-9:-6])
                grasp_dist = base_env.data.body_xpos[base_env.obj_bid].ravel()[-1]

                if d_i == 10:  # has an odd reach with different threshold
                    ep_reach_thresh = .062
                else:
                    ep_reach_thresh = relocate_reach_threshold

                ep_grasp_thresh = relocate_grasp_threshold

            if test_render:
                print(f"demo {d_i}, timestep: {t}, reach dist: {reach_dist:.4f}, grasp dist: {grasp_dist:.4f}")
                base_env.render()

            if reach_dist < ep_reach_thresh and not reached:

                if mode == 'reach':
                    expert_obs.append(obs)
                    if get_imgs: expert_imgs.append(base_env.render('rgb_array', width=img_wh[0], height=img_wh[1]))
                    num_ep_exp += 1
                if not allow_multiple: reached = True
                if mode == 'reach' and test_render:
                    import ipdb; ipdb.set_trace()

            if grasp_dist > ep_grasp_thresh and not grasped:
                if mode == 'grasp':
                    expert_obs.append(obs)
                    if get_imgs: expert_imgs.append(base_env.render('rgb_array', width=img_wh[0], height=img_wh[1]))
                    num_ep_exp += 1
                if not allow_multiple: grasped = True
                if mode == 'grasp' and test_render:
                    import ipdb; ipdb.set_trace()

            if t >= len(acts) - terminal_offset:
                if mode == 'final':
                    expert_obs.append(obs)
                    if get_imgs: expert_imgs.append(base_env.render('rgb_array', width=img_wh[0], height=img_wh[1]))
                    num_ep_exp += 1
                if mode == 'final' and test_render:
                    import ipdb; ipdb.set_trace()

        print(f"Num expert data in ep {d_i}: {num_ep_exp}")

        # import ipdb; ipdb.set_trace()

    # based on get_dataset methods from rce_envs.py
    print(f"Num expert data found: {len(expert_obs)}")
    num_obs = len(expert_obs)
    action_vec = [env.action_space.sample() for _ in range(num_obs)]
    dataset = {
        'observations': np.array(expert_obs, dtype=np.float32),
        'actions': np.array(action_vec, dtype=np.float32),
        'rewards': np.zeros(num_obs, dtype=np.float32),
    }

    if get_imgs:
        return dataset, np.array(expert_imgs)
    else:
        return dataset


def hand_dapg_reach_gen(base_env_class_inst, mode='reach'):
    # no noise for hand_dapg envs, since we only have human data
    AuxReachClass = type('AuxReachClass', (type(base_env_class_inst),), {})

    def aux_init(self):
        super(AuxReachClass, self).__init__()

    def get_dataset(self):
        return hand_dapg_get_dataset(self, base_env=base_env_class_inst, mode=mode)

    AuxReachClass.__init__ = aux_init
    AuxReachClass.get_dataset = get_dataset

    return AuxReachClass