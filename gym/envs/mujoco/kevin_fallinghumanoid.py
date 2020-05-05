import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from mujoco_py import functions as mjcf
import mujoco_py
#from mujoco_py import mjvisualize as mjcv

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    speed_weights = np.array([0, 10, 5, 3, .2, 5, 3, .2, 15, 25, 5, 3, 5, 3, 0])
    return (np.sum(mass * xpos[:,2] * speed_weights) / np.sum(mass) / np.sum(speed_weights))

class Kevin_FallingHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.counter = 0
        self.still_timer = 0
        self.begin_timer = 0

        self.force_weights = np.array([0, 1, 10, 4, 5, .1, .1, 10, 4, 5, .1, .1, 20, 20, 100, 20, 10, 5, 2, 10, 5, 2])

        dtr = 3.14159/180 #degrees to radians
        #                             free trans    free rot                       right leg           left leg            abdomen                      right arm        left arm
        self.init_qpos_low = np.array([0, 0, 0.85,   0.9950042, 0, 0.0998334, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   -35*dtr, -45*dtr, -30*dtr,   -85*dtr, -85*dtr, -90*dtr,   -60*dtr, -60*dtr, -90*dtr]) 
        self.init_qpos_high= np.array([0, 0, 0.85,   0.9950042, 0, 0.0998334, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   35*dtr , 45*dtr , 75*dtr ,   60*dtr , 60*dtr ,  50*dtr,   85*dtr , 85*dtr , 50*dtr])

        #                             free trans    free rot     right leg           left leg            abdomen    right arm  left arm
        self.init_qvel_low = np.array([0.4, 0, 0,   0, 0.8, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0]) 
        self.init_qvel_high= np.array([0.4, 0, 0,   0, 0.8, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0]) 

        mujoco_env.MujocoEnv.__init__(self, 'kevin_fallinghumanoid_pelvis.xml', 2)
        utils.EzPickle.__init__(self)
        print("Kevin Falling Humanoid environment set-up")

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat[10:-10],
                               data.cvel.flat[6:],
                               data.qfrc_actuator.flat[6:]]) #[np.array([6, 6, 9, 10, 12, 14, 15, 16])]])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        data = self.sim.data

        kin_energy_cost = 0.1 * np.sign(pos_after - pos_before) * np.square(pos_after - pos_before) / self.dt #kinetic energy is measured as the vertical displacement of the total CoM

        head_height_cost = 1 * min(data.body_xpos[14, 2]-0.3, 0) # A cost associated to keeping the head as high as possible

        quad_ctrl_cost = -0.1 * np.square(data.ctrl).sum()
        
        force_normals = np.zeros(self.force_weights.shape[0])
        for i in range(data.ncon):
            c_force = np.zeros(6, dtype=np.float64)
            mjcf.mj_contactForce(self.model, data, i, c_force)

            force_normals[data.contact[i].geom1] += c_force[0]
            force_normals[data.contact[i].geom2] += c_force[0]
        body_hit_cost = -4e-5 * np.sum(self.force_weights * force_normals) # Cost that is related to the impact force, with different weights for different body parts

        reward = kin_energy_cost + head_height_cost + quad_ctrl_cost + body_hit_cost

        #print("\rkin_energy_cost: {:f}  head_height_cost: {:f} body_hit_cost: {:f} quad_ctrl_cost: {:f} reward: {:f}".format(kin_energy_cost, head_height_cost, body_hit_cost, quad_ctrl_cost, reward), end="\n")

        if kin_energy_cost > 0:
            self.still_timer+= 1
        else:
            self.still_timer = 0
        self.begin_timer+=1

        done = (self.still_timer > 8 and self.begin_timer > 50)

        return self._get_obs(), reward, done, dict(reward_kin_energy=kin_energy_cost, reward_head_height=head_height_cost)

    def reset_model(self):
        c = 0.01 + self.counter*0.0001
        qpos = self.np_random.uniform(low=self.init_qpos_low, high=self.init_qpos_high) + self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        qvel = self.np_random.uniform(low=self.init_qvel_low, high=self.init_qvel_high) + self.np_random.uniform(low=-c, high=c, size=self.model.nv)

        self.still_timer = 0
        self.begin_timer = 0

        self.set_state(qpos, qvel)

        if self.counter < 500:
            self.counter+=0

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 1.0
        self.viewer.cam.elevation = -20

        '''
        #self.viewer.add_overlay()
        fncs = mjcf.__dict__
        #print(dir(mjcf)) 
        #print(dir(mjcf.mjr_figure))
        figure = mujoco_py.cymj.PyMjvFigure()
        con = mujoco_py.cymj.PyMjrContext()
        rect = mujoco_py.cymj.PyMjrRect()

        mujoco_py.cymj.MjRenderContextWindow()
        figure 
        #print(figure.__doc__())
        #print(dir(con)) 
        #print(dir(rect)) 
        #print(rect.width)

        mjcf.mjr_figure(rect, figure, con);
        
        #mjcf.mjv_defaultFigure(figure)
        raise SystemExit(0)
        '''