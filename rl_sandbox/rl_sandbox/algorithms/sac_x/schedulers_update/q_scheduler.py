import timeit
import torch
import numpy as np

import rl_sandbox.constants as c


class UpdateQScheduler:
    def __init__(self, model, algo_params):
        self.model = model
        self._num_tasks = algo_params.get(c.NUM_TASKS, 1)
        self._action_dim = algo_params[c.ACTION_DIM]

        self._scheduler_period = algo_params[c.SCHEDULER_SETTING][c.TRAIN][c.SCHEDULER_PERIOD]
        self._scheduler_tau = algo_params[c.SCHEDULER_TAU]
        self.main_intention = algo_params.get(c.MAIN_INTENTION, 0)

        self._gamma = algo_params[c.GAMMA]
        self._rewards = []
        self._discounting = []

    def state_dict(self):
        return self.model.state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _compute_returns(self):
        episode_length = len(self._rewards)
        returns = torch.zeros(episode_length + 1)
        for step in range(episode_length - 1, -1, -1):
            returns[step] = self._rewards[step] + \
                (self._gamma ** self._discounting[step]) * returns[step + 1]

        # Only take the returns for every scheduler's action
        return returns[:-1][::self._scheduler_period]

    def update_scheduler(self, obs, act, update_info):
        traj = obs + [act.item()]
        q_first_action = self.model.compute_qs([])

        print(f"Scheduler Trajectory: {traj} - Q([], a), for all a: {q_first_action}")

        tic = timeit.default_timer()
        update_info[c.Q_UPDATE_TIME] = []
        rets = self._compute_returns()
        for step in range(len(traj)):
            old_q_value = self.model.compute_qsa(traj[:step], traj[step])
            new_q_value = old_q_value * (1 - self._scheduler_tau) + rets[step] * self._scheduler_tau
            self.model.update_qsa(traj[:step], traj[step], new_q_value)
        update_info[c.Q_UPDATE_TIME].append(timeit.default_timer() - tic)
        update_info[c.SCHEDULER_TRAJ] = traj
        update_info[c.SCHEDULER_TRAJ_VALUE] = np.array(q_first_action)

    def update(self, obs, act, reward, done, info):
        self._rewards.append(reward[self.main_intention].item())
        self._discounting.append(info[c.DISCOUNTING][0].item())

        update_info = dict()
        if done:
            obs = info[c.HIGH_LEVEL_OBSERVATION]
            act = info[c.HIGH_LEVEL_ACTION]
            self.update_scheduler(obs, act, update_info)
            self._rewards.clear()
            self._discounting.clear()
            return True, update_info
        return False, update_info


class UpdateDACQScheduler(UpdateQScheduler):
    def __init__(self, model, reward_function, algo_params):
        super().__init__(model=model,
                         algo_params=algo_params)
        self.reward_function = reward_function
        self.max_ep_length = algo_params[c.MAX_EPISODE_LENGTH]
        self.curr_timestep = 0
        self.obss = []
        self.acts = []
        self.device = algo_params[c.DEVICE]
        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]
        self.main_intention = algo_params[c.MAIN_INTENTION]

    def _compute_returns(self):
        obss = self.train_preprocessing(torch.as_tensor(np.array(self.obss)).squeeze(1).float()).to(self.device)
        acts = torch.as_tensor(np.array(self.acts)).float().to(self.device)

        with torch.no_grad():
            rews = self.reward_function(obss, acts).detach()
        episode_length = len(rews)
        returns = torch.zeros(episode_length + 1)
        for step in range(episode_length - 1, -1, -1):
            returns[step] = rews[step, self.main_intention].cpu() + \
                (self._gamma ** self._discounting[step]) * returns[step + 1]

        self.obss = []
        self.acts = []

        # Only take the returns for every scheduler's action
        return returns[:-1][::self._scheduler_period]

    def update(self, obs, act, rew, done, info):
        # NOTE: We have entered absorbing state which is pretty much over...
        self.curr_timestep += 1
        self.obss.append(obs)
        self.acts.append(act)
        if obs[:, -1] == 1 or self.curr_timestep == self.max_ep_length:
            act[:] = 0
            done = True
            self.curr_timestep = 0

        return super().update(obs, act, rew, done, info)


class UpdateDACQSchedulerPlusHandcraft(UpdateQScheduler):
    def __init__(self, model, reward_function, algo_params):
        super().__init__(model=model,
                         algo_params=algo_params)
        self.reward_function = reward_function
        self.max_ep_length = algo_params[c.MAX_EPISODE_LENGTH]
        self.curr_timestep = 0
        self.obss = []
        self.acts = []
        self.device = algo_params[c.DEVICE]
        self.train_preprocessing = algo_params[c.TRAIN_PREPROCESSING]
        self.main_intention = algo_params[c.HANDCRAFT_TASKS]['main_task'][0]

    def _compute_returns(self):
        obss = self.train_preprocessing(torch.as_tensor(self.obss).squeeze(1).float()).to(self.device)
        acts = torch.as_tensor(self.acts).float().to(self.device)

        with torch.no_grad():
            rews = self.reward_function(obss, acts).detach()

        episode_length = len(rews)
        returns = torch.zeros(episode_length + 1)

        for step in range(episode_length - 1, -1, -1):
            returns[step] = rews[step, self.main_intention].cpu() + \
                (self._gamma ** self._discounting[step]) * returns[step + 1]

        self.obss = []
        self.acts = []

        # Only take the returns for every scheduler's action
        return returns[:-1][::self._scheduler_period]

    def update(self, obs, act, rew, done, info):
        # NOTE: We have entered absorbing state which is pretty much over...
        self.curr_timestep += 1
        self.obss.append(obs)
        self.acts.append(act)
        if obs[:, -1] == 1 or self.curr_timestep == self.max_ep_length:
            act[:] = 0
            done = True
            self.curr_timestep = 0

        return super().update(obs, act, rew, done, info)
