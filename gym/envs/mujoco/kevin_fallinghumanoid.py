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

        # variable which selects a test scenario if needed
        self.test = 0

        ###### Constants and ranges of initial positions and velocities ######
        # contact force weights
        self.force_weights = np.array([0, 1, 10, 2, 5, .1, .1, 10, 2, 5, .1, .1, 20, 20, 100, 20, 10, 5, 2, 10, 5, 2])

        dtr = math.pi/180 #degrees to radians
        # Initial free and joint positions, qpos[3:7] (rotation) are determined by qrot, so the quaternion can be declared properly
        #                             free trans    free rot       right leg           left leg            abdomen                      right arm                    left arm
        self.init_qpos_low = np.array([0, 0, 0.87,   1, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   -35*dtr, -45*dtr, -30*dtr,   -85*dtr, -85*dtr, -90*dtr,   -60*dtr, -60*dtr, -90*dtr]) 
        self.init_qpos_high= np.array([0, 0, 0.87,   1, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   35*dtr , 45*dtr , 75*dtr ,   60*dtr , 60*dtr ,  50*dtr,   85*dtr , 85*dtr , 50*dtr])
        #                             [rotation of fall, direction of fall (0=forward)]
        self.init_qrot_low = np.array([5*dtr, -15*dtr])
        self.init_qrot_high= np.array([15*dtr, 15*dtr])

        # Velocity of fall and initial joint velocities. qvel[0:6] is determined based upon qvel[0 & 3] and qrot[1] (direction of fall), so velocity is always in the direction of the fall
        #                             free trans    free rot     right leg           left leg            abdomen    right arm  left arm
        self.init_qvel_low = np.array([0.3, 0, 0,   0.6, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0]) 
        self.init_qvel_high= np.array([0.5, 0, 0,   1.0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0]) 
        ###### End of constants ######

        mujoco_env.MujocoEnv.__init__(self, 'kevin_fallinghumanoid_pelvis_stiff.xml', 2)
        utils.EzPickle.__init__(self)
        print("Kevin Falling Humanoid environment set-up")

    def _get_obs(self):
        data = self.sim.data
        '''
        # Realistic input
        return np.concatenate([data.qpos.flat[np.array([6, 8, 9, 10, 12, 14, 15, 16])],
                               data.qvel.flat[np.array([5, 7, 8, 9, 11, 13, 14, 15])],
                               data.sensordata,
                               -data.site_xmat.flat[np.array([2,5,8, 11,14,17, 20,23,26, 29,32,35, 38,41,44, 47,50,53])], 
                               [mjcf.mj_getTotalmass(self.model)],
                               data.qfrc_actuator.flat[np.array([6, 8, 9, 10, 12, 14, 15, 16])]])        
        '''
        # Original input
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat[10:-10],
                               data.cvel.flat[6:],
                               data.qfrc_actuator.flat[6:]]) #[np.array([6, 8, 9, 10, 12, 14, 15, 16])]])
        

    def get_plot_obs(self):
        data = self.sim.data

        impact_force = np.zeros(self.force_weights.shape[0])
        for i in range(data.ncon):
            c_force = np.zeros(6, dtype=np.float64)
            mjcf.mj_contactForce(self.model, data, i, c_force)
            impact_force[data.contact[i].geom1] += c_force[0]
            impact_force[data.contact[i].geom2] += c_force[0]
        impact_force[0]=0.0

        vert_vel = data.cvel[:,2]

        return impact_force, vert_vel

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        head_height_before = self.sim.data.body_xpos[14, 2]
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        data = self.sim.data

        kin_energy_cost = 2 * np.sign(pos_after - pos_before) * np.square(pos_after - pos_before) / self.dt #kinetic energy is measured as the vertical displacement of the total CoM

        head_height_cost = -0.2 * max(min(data.body_xpos[14, 2]-0.3, 0)*((data.body_xpos[14, 2]-head_height_before)/self.dt), 0) # A cost associated to keeping the head as high as possible

        quad_ctrl_cost = 0#-0.002 * np.square(data.ctrl).sum()
        
        force_normals = np.zeros(self.force_weights.shape[0])
        for i in range(data.ncon):
            c_force = np.zeros(6, dtype=np.float64)
            mjcf.mj_contactForce(self.model, data, i, c_force)
            #if c_force[0] > 5000: c_force[0]*=2
            force_normals[data.contact[i].geom1] += c_force[0]
            force_normals[data.contact[i].geom2] += c_force[0]
            #print("hit bodies are: {:d} and {:d} with force {:f}".format(data.contact[i].geom1, data.contact[i].geom2, c_force[0]), end="\n")
        body_hit_cost = -3e-6 * np.sum(self.force_weights * force_normals) # Cost that is related to the impact force, with different weights for different body parts

        reward = kin_energy_cost + head_height_cost + quad_ctrl_cost + body_hit_cost
        #reward = head_height_cost
        #print("\rkin_energy_cost: {:f} body_hit_cost: {:f} head_height_cost: {:f} quad_ctrl_cost: {:f} reward: {:f}".format(kin_energy_cost, body_hit_cost, head_height_cost, quad_ctrl_cost, reward), end="\n")
        
        if kin_energy_cost > -0.01:
            self.still_timer+= 1
        else:
            self.still_timer = 0
        self.begin_timer+=1

        done = False#(self.still_timer > 20 and self.begin_timer > 100)
        return self._get_obs(), reward, done, dict(reward_kin_energy=kin_energy_cost, reward_contact_force=body_hit_cost, reward_head_height=head_height_cost)

    def reset_model(self):
        c = 0.01

        qpos, qrot, qvel = self.select_init(c)

        self.still_timer = 0
        self.begin_timer = 0

        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 1.0
        self.viewer.cam.elevation = -20

    def select_init(self, c):
        if self.test == 0:
            mjcf.mj_setTotalmass(self.model, self.np_random.uniform(low=70, high=90))
            qpos = self.np_random.uniform(low=self.init_qpos_low, high=self.init_qpos_high) + self.np_random.uniform(low=-c, high=c, size=self.model.nq)
            qrot = self.np_random.uniform(low=self.init_qrot_low, high=self.init_qrot_high)
            qvel = self.np_random.uniform(low=self.init_qvel_low, high=self.init_qvel_high) + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        else:
            dtr = math.pi/180
            mjcf.mj_setTotalmass(self.model, 80)
            qpos = np.array([0, 0, 0.87,   1, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0])
            qrot = (self.init_qrot_low + self.init_qrot_high)/2
            qvel = (self.init_qvel_low + self.init_qvel_high)/2
            if self.test==2: #minimum mass
                mjcf.mj_setTotalmass(self.model, 70)
            elif self.test==3: # maximum mass
                mjcf.mj_setTotalmass(self.model, 90)
            elif self.test==4: #minimum initial push
                qrot[0] = self.init_qrot_low[0]
                qvel = self.init_qvel_low + 0
            elif self.test==5: #maximum initial push
                qrot[0] = self.init_qrot_high[0]
                qvel = self.init_qvel_high + 0
            elif self.test==6: #most towards the left
                qrot[1] = self.init_qrot_low[1]
            elif self.test==7: #most towards the right
                qrot[1] = self.init_qrot_high[1]
            elif self.test==8: #arms at the sides
                qpos = np.array([0, 0, 0.87,   1, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   50*dtr, -30*dtr, -90*dtr,   -50*dtr, 30*dtr, -90*dtr])
            elif self.test==9: #arms aimed backwards
                qpos = np.array([0, 0, 0.87,   1, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0,   0, 0, 0,   -85*dtr, -85*dtr, -90*dtr,   85*dtr, 85*dtr, -90*dtr])

        print("\rNew episode created with, Angle with ground: {:f}  Direction of fall: {:f}  Translational velocity: {:f}  Rotational velocity: {:f}  Mass: {:f}".format(qrot[0], qrot[1], qvel[0], qvel[3], mjcf.mj_getTotalmass(self.model)), end="\n")

        qpos[2] = qpos[2]*math.cos(qrot[0]*0.75)
        qpos[3:7] = euler_to_quaternion(qrot[0], qrot[1])
        self.model.qpos_spring[19:28] = qpos[19:28]
        qvel[0:6] = np.array([qvel[0]*math.cos(qrot[1]), -qvel[0]*math.sin(qrot[1]), 0, qvel[3]*math.sin(qrot[1]), qvel[3]*math.cos(qrot[1]), 0]) + self.np_random.uniform(low=-c, high=c, size=6)

        return qpos, qrot, qvel

    def add_to_test(self, value):
        self.test=(self.test+value)%10

    def get_test(self):
        return self.test
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