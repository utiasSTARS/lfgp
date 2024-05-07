import copy

import gym
import numpy as np
import transforms3d as tf3d

from d4rl.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
from d4rl.hand_manipulation_suite.door_v0 import DoorEnvV0
from d4rl.hand_manipulation_suite.hammer_v0 import HammerEnvV0


# Using functions instead of a generic DP class to try and maintain sanity with inheritance/super calls
def dp_step(self, a):
  if hasattr(self, 'delta_pos') and self.delta_pos and hasattr(self, "cur_act"):
    self.cur_act += a * self.delta_pos_multiplier
    self.cur_act = np.clip(self.cur_act, -1.0, 1.0)

    a = copy.deepcopy(self.cur_act)

  return a


def dp_reset_settle(self):
  self.cur_act = np.zeros(self.action_space.shape)

  # Allow env to settle
  num_settle_steps = int(200 / self.frame_skip * 5)
  for _ in range(num_settle_steps):
    self.step(np.zeros(self.action_space.shape))

  obs = self.get_obs()

  return obs


class DoorEnvV0DP(DoorEnvV0):
  def __init__(self,
               control_hz=20,
               common_control_multiplier=.02,
               responsive_control=False,
               lower_mass=True,
               delta_pos=True,
               **kwargs):
    super().__init__(**kwargs)

    self.delta_pos = delta_pos
    self.delta_pos_multiplier = common_control_multiplier * 20 / control_hz

    self.cur_act = np.zeros(self.action_space.shape)

    # increase frame skip to decrease control frequency -- max steps should be changed accordingly
    if control_hz == 5:
      self.frame_skip = 20

    t_idx = slice(self.sim.model.actuator_name2id('A_ARTz'), self.sim.model.actuator_name2id('A_ARTz')+1)
    r_idx = slice(self.sim.model.actuator_name2id('A_ARRx'), self.sim.model.actuator_name2id('A_ARRz')+1)

    if lower_mass:
      # reduce mass to improve control, forearm + all other hand links
      self.sim.model.body_mass[self.sim.model.body_name2id('forearm')] = 0.1
      self.sim.model.body_mass[self.sim.model.body_name2id('forearm') + 1:] *= 0.1

    if responsive_control:
      # change P param
      self.sim.model.actuator_gainprm[t_idx, 0] = np.ones(3) * 10000
      self.sim.model.actuator_biasprm[t_idx, 1] = np.ones(3) * (-4000)
      self.sim.model.actuator_gainprm[r_idx, 0] = np.ones(3) * 10000
      self.sim.model.actuator_biasprm[r_idx, 1] = np.ones(3) * (-4000)

  def step(self, a):
    a = dp_step(self, a)
    return super().step(a)

  def reset_model(self):
    super().reset_model()  # don't take the observation yet
    return dp_reset_settle(self)


class HammerEnvV0DP(HammerEnvV0):
  def __init__(self,
               control_hz=20,
               common_control_multiplier=.02,
               responsive_control=False,
               lower_mass=True,
               delta_pos=True,
               **kwargs):
    super().__init__(**kwargs)

    self.delta_pos = delta_pos
    self.delta_pos_multiplier = common_control_multiplier * 20 / control_hz

    self.cur_act = np.zeros(self.action_space.shape)

    # increase frame skip to decrease control frequency -- max steps should be changed accordingly
    if control_hz == 5:
      self.frame_skip = 20

    r_idx = slice(self.sim.model.actuator_name2id('A_ARRx'), self.sim.model.actuator_name2id('A_ARRy')+1)

    if lower_mass:
      # reduce mass to improve control, forearm + all other hand links
      self.sim.model.body_mass[self.sim.model.body_name2id('forearm')] = 0.1
      self.sim.model.body_mass[self.sim.model.body_name2id('forearm') + 1:] *= 0.1

    if responsive_control:
      # change P param
      self.sim.model.actuator_gainprm[r_idx, 0] = np.ones(3) * 10000
      self.sim.model.actuator_biasprm[r_idx, 1] = np.ones(3) * (-4000)

  def step(self, a):
    a = dp_step(self, a)
    return super().step(a)

  def reset_model(self):
    super().reset_model()  # don't take the observation yet
    return dp_reset_settle(self)


class RelocateEnvV0NoArmJointPosObs(RelocateEnvV0):
  """ A version of relocate using arm position instead of absolute joint position in obs """
  def __init__(self, include_vel=False, **kwargs):
    super().__init__(**kwargs)
    # super(RelocateEnvV0, self).__init__(**kwargs)
    # RelocateEnvV0.__init__(self, **kwargs)

    self.include_vel = include_vel

  def get_obs(self):
    # no hand position (implicit in last three subtractions), qpos for fingers, and site_xmat for hand rotation,
    # then converted to quat
    # xpos for obj
    # xpos for target
    qp = self.data.qpos.ravel()
    obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
    palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
    target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

    pm = self.data.site_xmat[self.S_grasp_sid].ravel()
    palm_rmat = np.array([[pm[0], pm[1], pm[2]], [pm[3], pm[4], pm[5]], [pm[6], pm[7], pm[8]]])
    palm_quat = tf3d.quaternions.mat2quat(palm_rmat)

    if hasattr(self, 'include_vel') and self.include_vel:
      p_vel = self.data.site_xvelp[self.S_grasp_sid].ravel()
      r_vel = self.data.site_xvelr[self.S_grasp_sid].ravel()
      p_obj_vel = self.data.body_xvelp[self.obj_bid].ravel()

      return np.concatenate([palm_quat, qp[6:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos,
                             p_vel, r_vel, p_obj_vel])

    else:
      return np.concatenate([palm_quat, qp[6:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])


class RelocateEnvV0DP(RelocateEnvV0):
  """ A delta-pos version of the relocate hand_dapg env. """
  def __init__(self,
               control_hz=20,
               common_control_multiplier=.02,
               responsive_control=False,
               rotate_frame_ee=True,
               lower_mass=True,
               delta_pos=True,
               **kwargs):
    super().__init__(**kwargs)

    self.delta_pos = delta_pos
    self.delta_pos_multiplier = common_control_multiplier * 20 / control_hz

    self.cur_act = np.zeros(self.action_space.shape)

    # increase frame skip to decrease control frequency -- max steps should be changed accordingly
    if control_hz == 5:
      self.frame_skip = 20

    t_idx = slice(self.sim.model.actuator_name2id('A_ARTx'), self.sim.model.actuator_name2id('A_ARTz')+1)
    r_idx = slice(self.sim.model.actuator_name2id('A_ARRx'), self.sim.model.actuator_name2id('A_ARRz')+1)

    if lower_mass:
      # reduce mass to improve control, forearm + all other hand links
      self.sim.model.body_mass[self.sim.model.body_name2id('forearm')] = 0.1
      self.sim.model.body_mass[self.sim.model.body_name2id('forearm') + 1:] *= 0.1

    if responsive_control:
      # change P param
      self.sim.model.actuator_gainprm[t_idx, 0] = np.ones(3) * 10000
      self.sim.model.actuator_biasprm[t_idx, 1] = np.ones(3) * (-4000)
      self.sim.model.actuator_gainprm[r_idx, 0] = np.ones(3) * 10000
      self.sim.model.actuator_biasprm[r_idx, 1] = np.ones(3) * (-4000)

      # if you want to reduce damping
      # self.sim.model.dof_damping[r_idx] = 2.0

    # manually change positions of rotational joints so we rotate about hand frame, instead of elbow
    if rotate_frame_ee:
      self.sim.model.jnt_pos[r_idx] = np.array([
        [ 0.  , -0.05,  0.5 ],
        [ 0.  , -0.05,  0.5 ],
        [ 0.  , -0.05,  0.5 ]
      ])

      # need to extend range towards table since it can't be reached in this case otherwise
      self.sim.model.jnt_range[t_idx][1, 0] = -0.13
      self.sim.model.actuator_ctrlrange[t_idx][1, 0] = -0.13

      self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
      self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

      if lower_mass:
        expert_min_norm = np.array([-.5, -1., -.4, -.4, -.35, -.1])
        expert_max_norm = np.array([.5, .5, .5, .3, .4, .25])
      else:
        expert_min_norm = np.array([-.5, 0.1, -.4, -.4, -.35, -.1])
        expert_max_norm = np.array([.5, .8, .5, .3, .4, .25])

    else:
      # change limits based on expert data to be more reasonable
      expert_min_norm = np.array([-.4, -1., -.3, -.4, -.35, -.1])
      expert_max_norm = np.array([.2, .1, .4, .3, .4, .25])

    expert_min = self.act_mid[:6] + expert_min_norm * self.act_rng[:6]
    expert_max = self.act_mid[:6] + expert_max_norm * self.act_rng[:6]

    self.act_mid[:6] = (expert_max + expert_min) / 2
    self.act_rng[:6] = (expert_max - expert_min) / 2

  def step(self, a):
    a = dp_step(self, a)
    return super().step(a)

  def reset_model(self):
    super().reset_model()  # don't take the observation yet
    return dp_reset_settle(self)


class RelocateEnvV0NoArmJointPosObsDP(RelocateEnvV0DP, RelocateEnvV0NoArmJointPosObs):
  """ A version of relocate no arm joint pos with delta pos """
  def __init__(self, **kwargs):
    # RelocateEnvV0NoArmJointPosObs.__init__(self, **kwargs)
    # RelocateEnvV0DP.__init__(self, **kwargs)

    super().__init__(**kwargs)