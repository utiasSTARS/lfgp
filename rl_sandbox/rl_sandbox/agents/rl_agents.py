import numpy as np
import torch

from torch.distributions import Categorical, Normal

import rl_sandbox.constants as c
from rl_sandbox.agents.random_agents import UniformContinuousAgent


class RLAgent():
    def __init__(self, model, learning_algorithm):
        self.model = model
        self.learning_algorithm = learning_algorithm

    def update(self, curr_obs, curr_h_state, action, reward, done, info, next_obs, next_h_state, **kwargs):
        return self.learning_algorithm.update(curr_obs,
                                              curr_h_state,
                                              action,
                                              reward,
                                              done,
                                              info,
                                              next_obs,
                                              next_h_state,
                                              **kwargs)

    def compute_action(self, obs, **kwargs):
        raise NotImplementedError

    def reset(self):
        # Returns initial hidden state
        if hasattr(self.model, c.INITIALIZE_HIDDEN_STATE):
            return self.model.initialize_hidden_state().numpy().astype(np.float32)
        return np.array([np.nan], dtype=np.float32)


class ACAgent(RLAgent):
    def __init__(self, model, learning_algorithm, preprocess=lambda obs: obs):
        super().__init__(model=model,
                         learning_algorithm=learning_algorithm)
        self.preprocess = preprocess

    def preprocess(self, obs):
        return obs

    def compute_action(self, obs, hidden_state):
        obs = torch.tensor(obs).unsqueeze(0)
        obs = self.preprocess(obs)
        hidden_state = torch.tensor(hidden_state).unsqueeze(0)
        action, value, hidden_state, log_prob, entropy, mean, variance = self.model.compute_action(
            obs, hidden_state)
        act_info = {c.VALUE: value,
                    c.LOG_PROB: log_prob,
                    c.ENTROPY: entropy,
                    c.MEAN: mean,
                    c.VARIANCE: variance}
        return action, hidden_state, act_info

    def deterministic_action(self, obs, hidden_state):
        obs = torch.tensor(obs).unsqueeze(0)
        obs = self.preprocess(obs)
        hidden_state = torch.tensor(hidden_state).unsqueeze(0)
        action, value, hidden_state, log_prob, entropy = self.model.deterministic_action(
            obs, hidden_state)
        act_info = {c.VALUE: value,
                    c.LOG_PROB: log_prob,
                    c.ENTROPY: entropy}
        return action, hidden_state, act_info


class ACAgentEUniformExplorer(ACAgent):
    """ Agent that enforces more exploration.

    prob_explore_ep: probability of executing an "exploration" episode. Determined during call to agent.reset().
    prob_explore_act: probablility of executing an exploratory action during exploration episode.
    max_repeat: max number of timesteps to repeat exploratory action.
    min_repeat: min number of timesteps to repeat exploratory action.
    """
    def __init__(self, model, learning_algorithm, prob_explore_ep, prob_explore_act, max_repeat, min_repeat,
                 min_action=-1, max_action=1, preprocess=lambda obs: obs):
        super().__init__(model, learning_algorithm, preprocess)
        self._prob_explore_ep = prob_explore_ep
        self._prob_explore_act = prob_explore_act
        self._max_repeat = max_repeat
        self._min_repeat = min_repeat
        self._explore_ep = False
        self._cur_explore_act = None
        self._act_repeat_ts = 0
        self._act_repeat_length = 0
        self._action_dim = self.model._action_dim
        self._uni_rand_agent = UniformContinuousAgent(np.ones(self._action_dim) * min_action,
                                                      np.ones(self._action_dim) * max_action)

    def compute_action(self, obs, hidden_state):
        explore_act = False
        if self._explore_ep:
            if self._cur_explore_act is not None:
                if self._act_repeat_ts < self._act_repeat_length:
                    explore_act = True
                    self._act_repeat_ts +=1
                else:
                    # reset action repeat explore
                    self._cur_explore_act = None
                    self._act_repeat_ts = 0

            if self._cur_explore_act is None:
                explore_act = np.random.rand() < self._prob_explore_act

                if explore_act:
                    self._cur_explore_act = list(self._uni_rand_agent.compute_action())
                    self._cur_explore_act[1] = hidden_state
                    self._cur_explore_act = tuple(self._cur_explore_act)
                    self._act_repeat_length = np.random.randint(self._min_repeat, self._max_repeat)
                    self._act_repeat_ts +=1

        if explore_act:
            return self._cur_explore_act
        else:
            return super().compute_action(obs, hidden_state)

    def reset(self):
        self._explore_ep = np.random.rand() < self._prob_explore_ep
        self._cur_explore_act = None
        return super().reset()