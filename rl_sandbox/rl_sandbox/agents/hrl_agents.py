import numpy as np
import torch

from torch.distributions import Categorical

import rl_sandbox.constants as c

from rl_sandbox.agents.rl_agents import ACAgent


class HierarchicalRLAgent(ACAgent):
    def __init__(self, high_level_model, low_level_model, learning_algorithm, preprocess=lambda obs: obs):
        self.high_level_model = high_level_model
        self.curr_high_level_obs = None
        self.curr_high_level_act = None
        self.curr_high_level_h_state = None
        super().__init__(model=low_level_model,
                         learning_algorithm=learning_algorithm,
                         preprocess=preprocess)

    def update(self, curr_obs, curr_h_state, action, reward, done, info, next_obs, next_h_state,
               update_intentions=True, update_scheduler=True, update_buffer=True, **kwargs):
        if update_scheduler:
            info[c.HIGH_LEVEL_OBSERVATION] = self.curr_high_level_obs
            info[c.HIGH_LEVEL_HIDDEN_STATE] = self.curr_high_level_h_state
            info[c.HIGH_LEVEL_ACTION] = self.curr_high_level_act
        return self.learning_algorithm.update(
            curr_obs, curr_h_state, action, reward, done, info, next_obs, next_h_state,
            update_intentions, update_scheduler, update_buffer, **kwargs)


class SACXAgent(HierarchicalRLAgent):
    def __init__(self, scheduler, intentions, learning_algorithm, scheduler_period, preprocess=lambda obs: obs):
        super().__init__(high_level_model=scheduler,
                         low_level_model=intentions,
                         learning_algorithm=learning_algorithm,
                         preprocess=preprocess)
        assert scheduler_period > 0
        self._scheduler_period = scheduler_period

    def compute_action(self, obs, hidden_state):
        if self._curr_timestep % self._scheduler_period == 0:
            if self.curr_high_level_obs is not None:
                self.curr_high_level_obs.append(self.curr_high_level_act.item())
            else:
                self.curr_high_level_obs = []

            self.curr_high_level_act, self.curr_high_level_value, self.curr_high_level_h_state, \
                self.curr_high_level_log_prob, self.curr_high_level_entropy, self.curr_high_level_mean, self.curr_high_level_variance = \
                    self.high_level_model.compute_action(self.curr_high_level_obs, torch.tensor(self.curr_high_level_h_state))
            high_level_act_info = {c.VALUE: self.curr_high_level_value,
                                   c.LOG_PROB: self.curr_high_level_log_prob,
                                   c.ENTROPY: self.curr_high_level_entropy,
                                   c.MEAN: self.curr_high_level_mean,
                                   c.VARIANCE: self.curr_high_level_variance}
        
            # print(f"Sched: timestep {self._curr_timestep} -- high level act: {self.curr_high_level_act}")

        action, hidden_state, act_info = super().compute_action(obs, hidden_state)
        act_info[c.LOG_PROB] = act_info[c.LOG_PROB][self.curr_high_level_act]
        act_info[c.VALUE] = act_info[c.VALUE][self.curr_high_level_act]
        act_info[c.ENTROPY] = act_info[c.ENTROPY][self.curr_high_level_act]
        act_info[c.MEAN] = act_info[c.MEAN][self.curr_high_level_act]
        act_info[c.VARIANCE] = act_info[c.VARIANCE][self.curr_high_level_act]

        self._curr_timestep += 1
        return action[self.curr_high_level_act], hidden_state, act_info

    def deterministic_action(self, obs, hidden_state):
        if self._curr_timestep % self._scheduler_period == 0:
            if self.curr_high_level_obs is not None:
                self.curr_high_level_obs.append(self.curr_high_level_act.item())
            else:
                self.curr_high_level_obs = []

            self.curr_high_level_act, self.curr_high_level_value, self.curr_high_level_h_state, \
                self.curr_high_level_log_prob, self.curr_high_level_entropy = \
                    self.high_level_model.deterministic_action(self.curr_high_level_obs, torch.tensor(self.curr_high_level_h_state))
            high_level_act_info = {c.VALUE: self.curr_high_level_value,
                                   c.LOG_PROB: self.curr_high_level_log_prob,
                                   c.ENTROPY: self.curr_high_level_entropy}

        action, hidden_state, act_info = super().deterministic_action(obs, hidden_state)
        act_info[c.LOG_PROB] = act_info[c.LOG_PROB][self.curr_high_level_act]
        act_info[c.VALUE] = act_info[c.VALUE][self.curr_high_level_act]
        act_info[c.ENTROPY] = act_info[c.ENTROPY][self.curr_high_level_act]

        # Use the action from the main task
        self._curr_timestep += 1
        return action[self.curr_high_level_act], hidden_state, act_info

    def reset(self):
        self._curr_timestep = 0
        self.curr_high_level_obs = None
        self.curr_high_level_h_state = np.nan
        self.curr_high_level_act = None
        self.curr_high_level_value = None
        self.curr_high_level_log_prob = None
        self.curr_high_level_entropy = None
        self.curr_high_level_mean = None
        self.curr_high_level_variance = None

        if hasattr(self.learning_algorithm, 'reset'):
            self.learning_algorithm.reset()

        return super().reset()


class SACXPlusForcedScheduleAgent(SACXAgent):
    def __init__(self, forced_schedule, scheduler, intentions, learning_algorithm, scheduler_period,
                 preprocess=lambda obs: obs):
        """
        forced_schedule should be a dictionary with optional integer entries corresponding to each
        possible aux task. if the scheduler chooses one of these entries at the initial timestep, the
        forced schedule will be followed from that point forward.

        each integer entry should have a sub-dictionary containing the desired schedule, which must, at least, contain
        an entry for 0 (corresponding to timestep 0). The entries of the sub-dictionary can either be an integer,
        corresponding to the action to use starting from that timestep key, or a 2-item tuple, containing the action to
        use followed by a 'k' or a 'd' for keep or discard in buffers (if being saved to buffers). If there is no tuple,
        it is assumed that all data should be kept.
        The tuple can instead be a 3-tuple, containing 3 lists, in which case the action should be sampled from the set
        of actions contained in the first list, with probabilities contained in the 2nd list, and the corresponding
        index in the third list contains whether to keep or discard data from the chosen action.

        e.g. {1: {0: 4, 45: 1}} --> if 1 was selected at the first timestep, this would instead select 4 until
        timestep 45, then switch to 1.
        e.g. {0: {0: (3, 'd'), 25: 0}} --> if 0 was selected, this would force selection of 3 instead, with an attached
        indicator to say discard all data until timestep 25.
        e.g. {0: {0: ([0, 1, 2, 3, 4, 5], [.15, .15, .25, .15, .15, .15], ['k', 'd', 'd', 'd', 'd', 'd']), 45: 0} -->
        if 0 was selected, instead sample any of 0-5 with probabilities in 2nd list, and only keep the data from the
        first action if 0 was selected.
        e.g.
        {
            2: {
                'trans_on_suc': True,
                'action_order': [4, 1, 3, 2, 0],
                'buffers': [[4, 1, 3, 2, 5], [1, 3, 2, 5], [3, 2, 5], [2], [0, 2]],
                'success_order': [4, 1, 3, 2, 0]
            },
            5: {
                'trans_on_suc': True,
                'action_order': [4, 1, 3, 5],
                'buffers': [[4, 1, 3, 2, 5], [1, 3, 2, 5], [3, 2, 5], [5]],
                'success_order': [4, 1, 3, 5]
            }
        }
        """

        super().__init__(scheduler, intentions, learning_algorithm, scheduler_period, preprocess=preprocess)
        self._forced_schedule = forced_schedule
        self._original_selected_action = None
        self._keep_current_action = True
        self._current_action_buffers = []
        self._trans_on_suc = False
        self._trans_on_suc_index = 0
        self._all_trans_on_suc_complete = False
        self._curr_intention_timestep = 0
        self._success_order = None
        self.curr_high_level_success_check = None

    def set_new_action(self, new_action):
        self.curr_high_level_act, self.curr_high_level_value, self.curr_high_level_h_state, \
        self.curr_high_level_log_prob, self.curr_high_level_entropy = \
            self.high_level_model.select_action(new_action, self.curr_high_level_obs,
                                                torch.tensor(self.curr_high_level_h_state))

    def get_new_action_from_tuple(self, new_action):
        if len(new_action) == 2:
            self._keep_current_action = new_action[1] == 'k'
            actual_action = new_action[0]
        elif len(new_action) == 3:
            actual_action_i = np.random.choice(new_action[0], p=new_action[1])
            index_in_action_list = new_action[0].index(actual_action_i)
            self._keep_current_action = new_action[2][index_in_action_list] == 'k'
            actual_action = actual_action_i

        return actual_action, self._keep_current_action

    def compute_action(self, obs, hidden_state, suc_for_trans=False):
        if self._original_selected_action is not None:
            if self._trans_on_suc:
                if suc_for_trans:
                    if self._trans_on_suc_index < \
                            len(self._forced_schedule[self._original_selected_action]['action_order']) - 1:
                        self.curr_high_level_obs.append(self.curr_high_level_act.item())
                        self._trans_on_suc_index += 1
                        new_action = \
                            self._forced_schedule[self._original_selected_action]['action_order'][self._trans_on_suc_index]
                        self._current_action_buffers = \
                            self._forced_schedule[self._original_selected_action]['buffers'][self._trans_on_suc_index]
                        prev_high_level_success_check = self.curr_high_level_success_check
                        self.curr_high_level_success_check = self._success_order[self._trans_on_suc_index] if \
                            self._success_order is not None else self.curr_high_level_act

                        print(f"Action {self.curr_high_level_act.item()} done based on success of "
                              f"{prev_high_level_success_check}, manually switching action to {new_action} with success "
                              f"based on {self.curr_high_level_success_check} keep actions in each of these buffers "
                              f"{self._current_action_buffers}")

                        self.set_new_action(new_action)
                        self._curr_intention_timestep = 0
                    else:
                        self._all_trans_on_suc_complete = True
            else:
                if self._curr_timestep in self._forced_schedule[self._original_selected_action]:

                    self.curr_high_level_obs.append(self.curr_high_level_act.item())

                    new_action = self._forced_schedule[self._original_selected_action][self._curr_timestep]
                    if type(new_action) == tuple:
                        new_action, _ = self.get_new_action_from_tuple(new_action)

                    else:
                        self._keep_current_action = True

                    print(f"Manually switching action to {new_action}, keep actions set to {self._keep_current_action}")

                    self.set_new_action(new_action)

        elif self._curr_timestep % self._scheduler_period == 0:
            if self.curr_high_level_obs is not None:
                self.curr_high_level_obs.append(self.curr_high_level_act.item())
            else:
                self.curr_high_level_obs = []

            self.curr_high_level_act, self.curr_high_level_value, self.curr_high_level_h_state, \
                self.curr_high_level_log_prob, self.curr_high_level_entropy, self.curr_high_level_mean, self.curr_high_level_variance = \
                    self.high_level_model.compute_action(self.curr_high_level_obs, torch.tensor(self.curr_high_level_h_state))
            high_level_act_info = {c.VALUE: self.curr_high_level_value,
                                   c.LOG_PROB: self.curr_high_level_log_prob,
                                   c.ENTROPY: self.curr_high_level_entropy,
                                   c.MEAN: self.curr_high_level_mean,
                                   c.VARIANCE: self.curr_high_level_variance}

            if self._curr_timestep == 0 and self.curr_high_level_act.item() in self._forced_schedule.keys():
                chosen_action = self.curr_high_level_act.item()
                self._trans_on_suc = self._forced_schedule[chosen_action].get('trans_on_suc', False)

                if self._trans_on_suc:
                    forced_action = self._forced_schedule[chosen_action]['action_order'][0]
                    self._success_order = self._forced_schedule[chosen_action].get('success_order', None)
                    self.curr_high_level_success_check = self._success_order[0] if self._success_order is not None \
                        else self.curr_high_level_act
                    new_keep_current_action = True  # unused
                    self._current_action_buffers = self._forced_schedule[chosen_action]['buffers'][0]

                    print(f"Selected action {chosen_action}, manually forcing action to {forced_action} with success "
                          f"check on {self.curr_high_level_success_check}, "
                          f"keep actions in each of these buffers {self._current_action_buffers}")
                else:
                    forced_action = self._forced_schedule[chosen_action][0]

                    if type(forced_action) == tuple:
                        forced_action, new_keep_current_action = self.get_new_action_from_tuple(forced_action)

                    else:
                        new_keep_current_action = True

                    print(f"Manually forcing action to {forced_action}, keep actions set to {new_keep_current_action}")
                self.reset()
                self.curr_high_level_obs = []
                self.set_new_action(forced_action)
                self._original_selected_action = chosen_action
                self._keep_current_action = new_keep_current_action

        action, hidden_state, act_info = super(SACXAgent, self).compute_action(obs, hidden_state)

        act_info[c.LOG_PROB] = act_info[c.LOG_PROB][self.curr_high_level_act]
        act_info[c.VALUE] = act_info[c.VALUE][self.curr_high_level_act]
        act_info[c.ENTROPY] = act_info[c.ENTROPY][self.curr_high_level_act]
        act_info[c.MEAN] = act_info[c.MEAN][self.curr_high_level_act]
        act_info[c.VARIANCE] = act_info[c.VARIANCE][self.curr_high_level_act]

        self._curr_timestep += 1
        self._curr_intention_timestep += 1
        return action[self.curr_high_level_act], hidden_state, act_info

    def reset(self):
        self._trans_on_suc_index = 0
        self._all_trans_on_suc_complete = False
        self._keep_current_action = True
        self._original_selected_action = None
        self._curr_intention_timestep = 0
        return super().reset()


class SACXPlusHandcraftAgent(HierarchicalRLAgent):
    def __init__(self, scheduler, intentions, learning_algorithm, scheduler_period, preprocess=lambda obs: obs):
        assert scheduler_period > 0
        self._scheduler_period = scheduler_period

        super().__init__(high_level_model=scheduler,
                         low_level_model=intentions,
                         learning_algorithm=learning_algorithm,
                         preprocess=preprocess)

        self._handcraft_task_is = []
        for task, task_i in self.model._handcraft_tasks.items():
            if task == 'main_task': pass
            elif task == 'open_action' or task == 'close_action':
                self._handcraft_task_is.append(task_i)
            else:
                raise NotImplementedError(f"Unrecognized task {task}")
        self._all_task_dim = self.model._task_dim + len(self._handcraft_task_is)

    def compute_action(self, obs, hidden_state):
        if self._curr_timestep % self._scheduler_period == 0:
            if self.curr_high_level_obs is not None:
                self.curr_high_level_obs.append(self.curr_high_level_act.item())
            else:
                self.curr_high_level_obs = []

            self.curr_high_level_act, self.curr_high_level_value, self.curr_high_level_h_state, \
                self.curr_high_level_log_prob, self.curr_high_level_entropy, self.curr_high_level_mean, self.curr_high_level_variance = \
                    self.high_level_model.compute_action(self.curr_high_level_obs, torch.tensor(self.curr_high_level_h_state))
            high_level_act_info = {c.VALUE: self.curr_high_level_value,
                                   c.LOG_PROB: self.curr_high_level_log_prob,
                                   c.ENTROPY: self.curr_high_level_entropy,
                                   c.MEAN: self.curr_high_level_mean,
                                   c.VARIANCE: self.curr_high_level_variance}

        action, hidden_state, act_info = super().compute_action(obs, hidden_state)

        # these are trash for this but it was too much work to fix it
        act_info[c.LOG_PROB] = act_info[c.LOG_PROB][0]
        act_info[c.VALUE] = act_info[c.VALUE][0]
        act_info[c.ENTROPY] = act_info[c.ENTROPY][0]
        act_info[c.MEAN] = act_info[c.MEAN][0]
        act_info[c.VARIANCE] = act_info[c.VARIANCE][0]
        act_info[c.VARIANCE] = act_info[c.VARIANCE][0]

        # act_info[c.LOG_PROB] = act_info[c.LOG_PROB][self.curr_high_level_act]
        # if self.curr_high_level_act in self._handcraft_task_is:
        #     act_info_index = np.array(self.model._main_task_original_i)
        # else:
        #     act_info_index = self.curr_high_level_act
        # if len(act_info[c.VALUE]) > 1:
        #     act_info[c.VALUE] = act_info[c.VALUE][act_info_index]
        #     act_info[c.ENTROPY] = act_info[c.ENTROPY][act_info_index]
        #     act_info[c.MEAN] = act_info[c.MEAN][act_info_index]
        #     act_info[c.VARIANCE] = act_info[c.VARIANCE][act_info_index]

        self._curr_timestep += 1
        return action[self.curr_high_level_act], hidden_state, act_info

    def deterministic_action(self, obs, hidden_state):
        if self._curr_timestep % self._scheduler_period == 0:
            if self.curr_high_level_obs is not None:
                self.curr_high_level_obs.append(self.curr_high_level_act.item())
            else:
                self.curr_high_level_obs = []

            self.curr_high_level_act, self.curr_high_level_value, self.curr_high_level_h_state, \
                self.curr_high_level_log_prob, self.curr_high_level_entropy = \
                    self.high_level_model.deterministic_action(self.curr_high_level_obs, torch.tensor(self.curr_high_level_h_state))
            high_level_act_info = {c.VALUE: self.curr_high_level_value,
                                   c.LOG_PROB: self.curr_high_level_log_prob,
                                   c.ENTROPY: self.curr_high_level_entropy}

        action, hidden_state, act_info = super().deterministic_action(obs, hidden_state)

        # these are trash for this but it was too much work to fix it
        act_info[c.LOG_PROB] = act_info[c.LOG_PROB][0]
        act_info[c.VALUE] = act_info[c.VALUE][0]
        act_info[c.ENTROPY] = act_info[c.ENTROPY][0]

        # act_info[c.LOG_PROB] = act_info[c.LOG_PROB][self.curr_high_level_act]
        # act_info[c.VALUE] = act_info[c.VALUE][self.curr_high_level_act]
        # act_info[c.ENTROPY] = act_info[c.ENTROPY][self.curr_high_level_act]

        # Use the action from the main task
        self._curr_timestep += 1
        return action[self.curr_high_level_act], hidden_state, act_info

    def reset(self):
        self._curr_timestep = 0
        self.curr_high_level_obs = None
        self.curr_high_level_h_state = np.nan
        self.curr_high_level_act = None
        self.curr_high_level_value = None
        self.curr_high_level_log_prob = None
        self.curr_high_level_entropy = None
        self.curr_high_level_mean = None
        self.curr_high_level_variance = None
        return super().reset()

class DIAYNAgent(HierarchicalRLAgent):
    """ One may consider sampling skill from prior distribution as a high level action
    """
    def __init__(self, prior, model, learning_algorithm, preprocess=lambda obs: obs):
        super().__init__(high_level_model=prior,
                         low_level_model=model,
                         learning_algorithm=learning_algorithm,
                         preprocess=preprocess)

    def compute_action(self, obs, hidden_state):
        return super().compute_action(obs, hidden_state)

    def reset(self):
        self.curr_high_level_act = self.high_level_model.sample(num_samples=(1,))
        return super().reset()
