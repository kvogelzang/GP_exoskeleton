import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class Kevin_HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'kevin_humanoid.xml', 2)
        utils.EzPickle.__init__(self)
        print("Kevin environment set-up")

    def _get_obs(self):
        data = self.sim.data
        '''
        print("qpos")
        print(data.qpos)
        print("qvel")
        print(data.qvel)
        print("cvel")
        print(data.cvel.flat[6:])

        print("qfrc_actuator")
        print(data.qfrc_actuator.flat[np.array([9, 11, 12, 13, 15, 17, 18, 19])])   
        
        print("cinert")
        print(data.cinert)
        print(data.xipos)
        print(data.ximat)
        raise SystemExit(0)
        ''' 
        #np.concatenate([data.qpos.flat[2:], data.qvel.flat, data.cinert.flat, data.cvel.flat, data.qfrc_actuator.flat, data.cfrc_ext.flat])
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat[10:],
                               data.cvel.flat[6:],
                               data.qfrc_actuator.flat[np.array([9, 11, 12, 13, 15, 17, 18, 19])]])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        qpos = self.sim.data.qpos
        alive_bonus = 2.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        height_cost = - 3 * (min(qpos[2] - 1.25, 0))
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        reward = lin_vel_cost - height_cost - quad_ctrl_cost + alive_bonus
        #print("\rlin_vel_cost: {:f}  quad_ctrl_cost: {:f} height_cost: {:f} reward: {:f}".format(lin_vel_cost, quad_ctrl_cost, height_cost, reward), end="\n")

        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        #reward += -done*20
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 1.0
        self.viewer.cam.elevation = -20
