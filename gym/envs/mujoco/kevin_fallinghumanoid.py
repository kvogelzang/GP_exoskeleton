import numpy as np
import math
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

def euler_to_quaternion(rot, theta):
        roll = rot*math.cos(theta)
        pitch = rot*math.sin(theta)

        qw = np.cos(roll/2) * np.cos(pitch/2)
        qx = np.cos(roll/2) * np.sin(pitch/2)
        qy = np.sin(roll/2) * np.cos(pitch/2)
        qz = - np.sin(roll/2) * np.sin(pitch/2)

        return [qw, qx, qy, qz]

class Kevin_FallingHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # timers used to determine end of episode, declared here so they carry over to next timestep
        self.still_timer = 0
        self.begin_timer = 0

        ###### Constants and ranges of initial positions and velocities ######
        # contact force weights
        self.force_weights = np.array([0, 1, 10, 4, 5, .1, .1, 10, 4, 5, .1, .1, 20, 20, 100, 20, 10, 5, 2, 10, 5, 2])

        dtr = math.pi/180 #degrees to radians
        # Initial free and joint positions, qpos[3:7] (rotation) are determined by qrot, so the quaternion can be declared properly
        #                             free trans    free rot       right leg           left leg            abdomen                      right arm        left arm
        self.init_qpos_low = np.array([0, 0, 0.87,   1, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   -35*dtr, -45*dtr, -30*dtr,   -85*dtr, -85*dtr, -90*dtr,   -60*dtr, -60*dtr, -90*dtr]) 
        self.init_qpos_high= np.array([0, 0, 0.87,   1, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   35*dtr , 45*dtr , 75*dtr ,   60*dtr , 60*dtr ,  50*dtr,   85*dtr , 85*dtr , 50*dtr])
        #                             [rotation of fall, direction of fall (0=forward)]
        self.init_qrot_low = np.array([5*dtr, -5*dtr])
        self.init_qrot_high= np.array([15*dtr, 5*dtr])

        # Velocity of fall and initial joint velocities. qvel[0:6] is determined based upon qvel[0 & 3] and qrot[1] (direction of fall), so velocity is always in the direction of the fall
        #                             free trans    free rot     right leg           left leg            abdomen    right arm  left arm
        self.init_qvel_low = np.array([0.2, 0, 0,   0.4, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0]) 
        self.init_qvel_high= np.array([0.5, 0, 0,   1.0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0]) 
        ###### End of constants ######

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
        c = 0.01
        mjcf.mj_setTotalmass(self.model, self.np_random.uniform(low=70, high=90))

        qpos = self.np_random.uniform(low=self.init_qpos_low, high=self.init_qpos_high) + self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        qrot = self.np_random.uniform(low=self.init_qrot_low, high=self.init_qrot_high)
        qpos[2] = qpos[2]*math.cos(qrot[0]*0.75)
        qpos[3:7] = euler_to_quaternion(qrot[0], qrot[1])

        qvel = self.np_random.uniform(low=self.init_qvel_low, high=self.init_qvel_high) + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        print("\r New episode created with, Angle with ground: {:f}  Direction of fall: {:f}  Translational velocity: {:f}  Rotational velocity: {:f}  Mass: {:f}".format(qrot[0], qrot[1], qvel[0], qvel[3], mjcf.mj_getTotalmass(self.model)), end="\n")

        qvel[0:6] = np.array([qvel[0]*math.cos(qrot[1]), -qvel[0]*math.sin(qrot[1]), 0, qvel[3]*math.sin(qrot[1]), qvel[3]*math.cos(qrot[1]), 0]) + self.np_random.uniform(low=-c, high=c, size=6)

        self.still_timer = 0
        self.begin_timer = 0

        self.set_state(qpos, qvel)

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