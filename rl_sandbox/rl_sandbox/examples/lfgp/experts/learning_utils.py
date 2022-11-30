import numpy as np
from collections import OrderedDict
from functools import partial
import copy

import rl_sandbox.constants as c
from rl_sandbox.buffers.utils import make_buffer
from rl_sandbox.agents.hrl_agents import SACXPlusForcedScheduleAgent


def multi_buffer_warmup(agent,
                        env,
                        buffers,
                        buffer_preprocess,
                        transition_preprocess,
                        experiment_settings,
                        reset_between_intentions=False,
                        pol_i_to_buffer_i=None,
                        render=False,
                        store_success_only=False,
                        reset_on_success=False):
    clip_action = experiment_settings.get(
        c.CLIP_ACTION, c.DEFAULT_TRAIN_PARAMS[c.CLIP_ACTION])
    min_action = experiment_settings.get(c.MIN_ACTION, None)
    max_action = experiment_settings.get(c.MAX_ACTION, None)
    num_steps = experiment_settings.get(c.NUM_STEPS, 0)
    num_episodes = experiment_settings.get(c.NUM_EPISODES, 0)
    num_episodes_per_buffer = experiment_settings.get("num_episodes_per_buffer", None)
    completed_episodes_per_sched_task = OrderedDict()
    num_episodes_per_buffer_dict = dict()

    if num_episodes_per_buffer is not None:
        for i, intention_i in enumerate(agent.high_level_model.task_options):  # only works for UScheduler...should be okay?
            completed_episodes_per_sched_task[intention_i] = 0
            num_episodes_per_buffer_dict[intention_i] = num_episodes_per_buffer[i]

    buffer_preprocess.reset()
    curr_obs = env.reset()
    curr_obs = buffer_preprocess(curr_obs)
    curr_h_state = agent.reset()
    curr_step = 0
    ep_step = 0
    curr_episode = 0

    success = False
    last_valid_step_aux_buffers = [0] * len(buffers)
    main_task_suc_all_trans_fail_timeout = 10  # timeout if main tasks succeeds without going through all transitions
    main_task_suc_all_trans_fail_count = 0
    successes = [False] * len(buffers)

    multi_buffers_per_timestep = hasattr(agent, '_current_action_buffers')

    if store_success_only or reset_on_success:
        assert hasattr(env, 'get_task_successes')
        experiment_settings[c.AUXILIARY_REWARDS].set_aux_rewards_str()
        aux_suc = partial(env.get_task_successes, tasks=experiment_settings[c.AUXILIARY_REWARDS]._aux_rewards_str)

        ep_buf_config = copy.deepcopy(experiment_settings[c.BUFFER_SETTING])
        ep_buf_config[c.BUFFER_WRAPPERS] = {}  # so that it isn't a torch buffer

        ep_bufs = [make_buffer(ep_buf_config) for _ in range(len(buffers))]

    while True:
        if hasattr(env, c.RENDER) and render:
            env.render()

        if type(agent) == SACXPlusForcedScheduleAgent:
            if ep_step == 0:
                suc_for_trans = False
            else:
                suc_for_trans = successes[agent.curr_high_level_act.item()] if \
                    agent.curr_high_level_success_check is None else successes[agent.curr_high_level_success_check]
            if suc_for_trans:
                for i, buf in enumerate(ep_bufs):
                    last_valid_step_aux_buffers[i] = buf._count

            action, next_h_state, act_info = agent.compute_action(
                obs=curr_obs, hidden_state=curr_h_state, suc_for_trans=suc_for_trans)
        else:
            action, next_h_state, act_info = agent.compute_action(obs=curr_obs, hidden_state=curr_h_state)

        env_action = action
        if clip_action:
            env_action = np.clip(
                action, a_min=min_action, a_max=max_action)

        next_obs, reward, done, env_info = env.step(env_action)
        next_obs = buffer_preprocess(next_obs)

        if type(agent) == SACXPlusForcedScheduleAgent and agent._original_selected_action is not None:
            selected_action_i = agent._original_selected_action
            store_in_buffer = agent._keep_current_action
        else:
            selected_action_i = agent.curr_high_level_act.item()
            store_in_buffer = True

        if store_success_only or reset_on_success:
            successes = aux_suc(observation=curr_obs, action=action, env_info=env_info['infos'][-1])
            if type(agent) == SACXPlusForcedScheduleAgent and agent._original_selected_action is not None:
                success = successes[agent._original_selected_action] and \
                    agent.curr_high_level_act.item() == agent._original_selected_action
            else:
                success = successes[selected_action_i]

            if pol_i_to_buffer_i is None:  # need to set ep_buf even if we're not storing anything
                curr_buf = buffers[selected_action_i]
                ep_buf = ep_bufs[selected_action_i]
            else:
                ep_buf = ep_bufs[pol_i_to_buffer_i[selected_action_i]]
                curr_buf = buffers[pol_i_to_buffer_i[selected_action_i]]

        info = dict()
        info[c.DISCOUNTING] = env_info.get(c.DISCOUNTING, 1)
        info.update(act_info)

        if multi_buffers_per_timestep and agent._trans_on_suc:
            for pol_i in agent._current_action_buffers:
                buf = ep_bufs[pol_i_to_buffer_i[pol_i]]  # pol_i_to_buffer_i is definitely not None in this case
                buf.push(**transition_preprocess(curr_obs, curr_h_state, action, reward, done, info,
                                                 next_obs=next_obs, next_h_state=next_h_state))
        else:
            if pol_i_to_buffer_i is None:
                curr_buf = buffers[selected_action_i]
            else:
                curr_buf = buffers[pol_i_to_buffer_i[selected_action_i]]

            if len(curr_buf) < curr_buf._memory_size and not store_success_only and store_in_buffer:
                curr_buf.push(**transition_preprocess(curr_obs, curr_h_state, action, reward, done, info,
                                                      next_obs=next_obs, next_h_state=next_h_state))

            if store_success_only and store_in_buffer:
                if pol_i_to_buffer_i is None:
                    ep_buf = ep_bufs[selected_action_i]
                else:
                    ep_buf = ep_bufs[pol_i_to_buffer_i[selected_action_i]]
                ep_buf.push(**transition_preprocess(curr_obs, curr_h_state, action, reward, done, info,
                                                    next_obs=next_obs, next_h_state=next_h_state))

        curr_obs = next_obs
        curr_h_state = next_h_state
        curr_step += 1
        ep_step += 1

        if reset_between_intentions:
            if multi_buffers_per_timestep and agent._trans_on_suc:
                if agent._curr_intention_timestep % agent._scheduler_period == 0:
                    done = True
            else:
                if ep_step % agent._scheduler_period == 0:
                    done = True

        else:
            if reset_on_success:
                # means we're collecting play data, but want to switch on timeout OR success, not just timeout
                if success:
                    print(f"Task {selected_action_i} succeeded! saving to buffer.")
                    agent._curr_timestep = 0  # forces new action selection
                    if len(curr_buf) < curr_buf._memory_size and success and len(ep_buf) > 0:
                        o, h, a, r, d, no, nh, i, l, _ = ep_buf.sample_with_next_obs(None, None, None,
                                                                                     np.array(range(len(ep_buf))))
                        curr_buf.push_multiple(obss=o, h_states=h, acts=a, rews=r, dones=d, infos=i, next_obss=no,
                                               next_h_states=nh)
                    buf_lens = np.array([len(buf) for buf in buffers])
                    ep_bufs = [make_buffer(ep_buf_config) for _ in range(len(buffers))]
                    success = False
                    env.reset_episode_success_data()

        if reset_on_success:
            if success:
                if multi_buffers_per_timestep and agent._trans_on_suc:
                    if agent._all_trans_on_suc_complete:
                        done = True
                    else:

                        # if a specific intention in 'success_order' never reaches success, that data
                        # can still be stored in the main buffer as long as success is met, but can't be stored
                        # in any of the aux buffers (starting from and including whatever was supposed to be stored
                        # at that step), because we don't know if and when transitions have happened
                        if successes[agent._original_selected_action]:
                            main_task_suc_all_trans_fail_count += 1

                            if main_task_suc_all_trans_fail_count >= main_task_suc_all_trans_fail_timeout:
                                print("Main task succeeded, but aux task transitions failed, so keeping "
                                      "all main task data but discarding invalid aux task data.")
                                done = True
                                for pol_i in pol_i_to_buffer_i.keys():
                                    if pol_i == agent._original_selected_action:
                                        continue
                                    buf = ep_bufs[pol_i_to_buffer_i[pol_i]]
                                    buf._count = last_valid_step_aux_buffers[pol_i_to_buffer_i[pol_i]]
                            else:
                                done = False
                else:
                    done = True

        buf_lens = np.array([len(buf) for buf in buffers])

        if done:
            if store_success_only:
                if num_episodes_per_buffer is not None:
                    if success and completed_episodes_per_sched_task[agent._original_selected_action] < \
                            num_episodes_per_buffer_dict[agent._original_selected_action]:
                        completed_episodes_per_sched_task[agent._original_selected_action] += 1

                if multi_buffers_per_timestep and agent._trans_on_suc:
                    for buf_i, buf in enumerate(ep_bufs):
                        if num_episodes_per_buffer is not None:
                            intention = agent._original_selected_action
                            keep_buf = completed_episodes_per_sched_task[intention] <= \
                                           num_episodes_per_buffer_dict[intention] and success and len(buf) > 0
                        else:
                            keep_buf = len(buffers[buf_i]) < buffers[buf_i]._memory_size and success and len(buf) > 0
                        if keep_buf:
                            o, h, a, r, d, no, nh, i, l, _ = buf.sample_with_next_obs(None, None, None,
                                                                                         np.array(range(len(buf))))
                            buffers[buf_i].push_multiple(obss=o, h_states=h, acts=a, rews=r, dones=d, infos=i,
                                                         next_obss=no, next_h_states=nh)
                else:
                    if len(curr_buf) < curr_buf._memory_size and success and len(ep_buf) > 0:
                        o, h, a, r, d, no, nh, i, l, _ = ep_buf.sample_with_next_obs(None, None, None, np.array(range(len(ep_buf))))
                        curr_buf.push_multiple(obss=o, h_states=h, acts=a, rews=r, dones=d, infos=i, next_obss=no, next_h_states=nh)
                buf_lens = np.array([len(buf) for buf in buffers])
                ep_bufs = [make_buffer(ep_buf_config) for _ in range(len(buffers))]

            print("Intention buffer lengths: ", buf_lens)
            buffer_preprocess.reset()
            curr_obs = env.reset()
            curr_obs = buffer_preprocess(curr_obs)
            curr_h_state = agent.reset()
            curr_episode += 1
            ep_step = 0

            main_task_suc_all_trans_fail_count = 0
            last_valid_step_aux_buffers = [0] * len(buffers)
            success = False
            successes = [False] * len(buffers)

            if curr_episode >= num_episodes:
                break

            if num_episodes_per_buffer is not None:
                all_done = []
                completed_episodes = []
                tasks = []
                for k in completed_episodes_per_sched_task.keys():
                    all_done.append(completed_episodes_per_sched_task[k] >= num_episodes_per_buffer_dict[k])
                    completed_episodes.append(completed_episodes_per_sched_task[k])
                    tasks.append(k)
                episodes_per_buffer_str = [str(task) + ": " + str(num) for task, num in zip(tasks, completed_episodes)]
                print(f"Episodes per buffer: {episodes_per_buffer_str}")
                if all(all_done):
                    break

        bufs_done = [len(buf) == buf._memory_size for buf in buffers]
        if all(bufs_done):
            break