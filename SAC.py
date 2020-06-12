import math
import random

import gym
import numpy as np
from numpy import inf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt

import datetime
import pickle
import os

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        discount_factor = 0.98
        self.time_horizon = 20
        self.discounted_reward = np.ones(self.time_horizon)
        for i in range(self.time_horizon):
            self.discounted_reward[i]*=discount_factor**i
        sum_weights = sum(self.discounted_reward)
        for i in range(self.time_horizon):
            self.discounted_reward[i]/=sum_weights
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        #batch = self.buffer
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        '''
        # This can be used to plot a subset of the data
        length = 400
        plot(len(self.buffer), action[self.position-length:self.position, :], "Normalized action (-1,1)", legend=1)
        plot(len(self.buffer), np.transpose([reward[self.position-length:self.position], done[self.position-length:self.position]==True]), "Reward")
        #plot(len(self.buffer), state[self.position-length:self.position, 0:5], "Free body position [0=m] / [1:4 = quat]", legend=True)
        #plot(len(self.buffer), state[self.position-length:self.position, 5:26], "Joint position [rad]")
        #plot(len(self.buffer), state[self.position-length:self.position, 26:32], "Free body velocity [0:2=m/s] / [3:5=rad/s]", legend=True)
        #plot(len(self.buffer), state[self.position-length:self.position, 32:53], "Joint velocity [rad/s]")
        #plot(len(self.buffer), state[self.position-200:self.position, 53:61])
        #for i in range(10, 14):
        #    print(i)
        #    plot(len(self.buffer), state[self.position-200:self.position, 53+10*i:63+10*i])
        #    plot(len(self.buffer), state[self.position-200:self.position, 183+6*i:189+6*i])
        plot(len(self.buffer), state[self.position-length:self.position, np.array([267,269,270,271,273,275,276,277])], "Actuator torque [Nm/rad]", legend=1) #78:106, 267:289 np.array([267,269,270,271,273,275,276,277])
        raise SystemExit(0)
        '''
        return state, action, reward, next_state, done
    
    def insert(self, temp_buffer, old_rewards):
        old_rewards = np.array(old_rewards)
        new_rewards = []
        prev_action = np.zeros(action_dim)

        for i in range(len(temp_buffer)):
            if i+self.time_horizon<len(temp_buffer):
                new_rewards.append(sum(old_rewards[i:i+self.time_horizon]*self.discounted_reward))
            else:
                new_rewards.append(sum(old_rewards[i:]*self.discounted_reward[0:len(temp_buffer)-i])/sum(self.discounted_reward[0:len(temp_buffer)-i]))

            state, action, _, next_state, done = temp_buffer.buffer[i]
            new_rewards[-1]+=-0.05*np.square(action).sum()
            #new_rewards[-1]+=-0.025*np.square(torch.FloatTensor(prev_action).to(device) - action).sum()
            self.push(state, action, new_rewards[-1], next_state, done)
            prev_action = action

        temp_buffer.buffer = []
        temp_buffer.position = 0

        return new_rewards

    def __len__(self):
        return len(self.buffer)

def plot(frame_idx, rewards, ylabel="", xlabel=None, subplot=None, legend=None):
    clear_output(True)
    plt.figure(figsize=(10,5))
    if subplot:
        plt.subplot(subplot)
    #plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(range(len(rewards)), rewards, marker='.', ms=1)
    plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("Past time steps (10 ms per step)")
    if legend:
        plt.legend(range(rewards.shape[1]))
    plt.show()

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, norm_weights, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        drawfrom = np.transpose(np.repeat(np.expand_dims((1/np.sqrt(state_dim))*norm_weights, axis=1), hidden_dim, axis=1))
        self.linear1.weight.data = torch.FloatTensor(np.random.uniform(-drawfrom, drawfrom))
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, norm_weights, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        drawfrom = np.transpose(np.repeat(np.expand_dims((1/np.sqrt(num_inputs + num_actions))*np.concatenate([norm_weights, np.ones(num_actions)]), axis=1), hidden_size, axis=1))
        self.linear1.weight.data = torch.FloatTensor(np.random.uniform(-drawfrom, drawfrom))
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x
    
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, norm_weights, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        drawfrom = np.transpose(np.repeat(np.expand_dims((1/np.sqrt(state_dim))*norm_weights, axis=1), hidden_size, axis=1))
        self.linear1.weight.data = torch.FloatTensor(np.random.uniform(-drawfrom, drawfrom))
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(torch.FloatTensor(np.zeros(mean.shape[1])), torch.FloatTensor(np.ones(std.shape[1])))
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(device)) # Reparametrization trick
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob, z, mean, log_std
        
    def get_action(self, state, random=1):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(torch.FloatTensor(np.zeros(mean.shape[1])), torch.FloatTensor(random*np.ones(std.shape[1])))
        z      = normal.sample().to(device)
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()#.detach().cpu().numpy()
        return action[0]
        '''
    def show(self, state, action):
        # This can be used to show the activation of the various layers

        y = state #torch.cat([state, action], 1)
        y = np.repeat(np.expand_dims(y, axis=1), self.linear1.weight.data.shape[0], axis=1)
        x = np.absolute(self.linear1.weight.data*y)
        x = np.max(x.numpy(), axis=0, keepdims=False)
        plt.figure()
        plt.imshow(np.transpose(x), cmap='gray')
        plt.show()
        raise SystemExit(0)
        '''

def update(batch_size,soft_tau=1e-2, alpha = 1):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    #policy_net.show(state, action)
    #raise SystemExit(0)
    #print(np.transpose(target_value_func.detach().numpy()))

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)

    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)
    log_prob = torch.mean(log_prob, 1, keepdim=True) #torch.log(torch.prod(log_prob.exp(), 1, keepdim=True))
    
    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    
    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()    

    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - alpha * log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())
    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    # Training Policy Function
    policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

def save(filename, directory):
    path = '../%s/%s' % (directory, directory_name)
    torch.save(value_net.state_dict(), '%s/%s_value_net.pth' % (path, filename))
    torch.save(target_value_net.state_dict(), '%s/%s_target_value_net.pth' % (path, filename))
    torch.save(soft_q_net1.state_dict(), '%s/%s_q1_net.pth' % (path, filename))
    torch.save(soft_q_net2.state_dict(), '%s/%s_q2_net.pth' % (path, filename))
    torch.save(policy_net.state_dict(), '%s/%s_policy_net.pth' % (path, filename))

    with open(path + "/" + filename + "_replay_buffer", 'wb') as pickle_file:
        pickle.dump(replay_buffer, pickle_file, pickle.HIGHEST_PROTOCOL)
    with open(path + "/" + filename + "_rewards", 'wb') as pickle_file:
        pickle.dump(rewards, pickle_file, pickle.HIGHEST_PROTOCOL)
    with open(path + "/" + filename + "_test_rewards", 'wb') as pickle_file:
        pickle.dump(test_rewards, pickle_file, pickle.HIGHEST_PROTOCOL)
    with open(path + "/" + filename + "_frame_idx", 'wb') as pickle_file:
        pickle.dump(frame_idx, pickle_file, pickle.HIGHEST_PROTOCOL)

def load(filename="end", directory="saves"):
    path = '../%s/%s' % (directory, directory_name)
    value_net.load_state_dict(torch.load('%s/%s_value_net.pth' % (path, filename)))
    target_value_net.load_state_dict(torch.load('%s/%s_target_value_net.pth' % (path, filename)))

    soft_q_net1.load_state_dict(torch.load('%s/%s_q1_net.pth' % (path, filename)))
    soft_q_net2.load_state_dict(torch.load('%s/%s_q2_net.pth' % (path, filename)))
    policy_net.load_state_dict(torch.load('%s/%s_policy_net.pth' % (path, filename)))

    if TRAIN!=2:
        with open(path + "/" + filename + "_replay_buffer", 'rb') as pickle_file:
            replay_buffer = pickle.load(pickle_file)
    else:
        replay_buffer = None

    with open(path + "/" + filename + "_rewards", 'rb') as pickle_file:
        rewards = pickle.load(pickle_file)

    try:
        with open(path + "/" + filename + "_test_rewards", 'rb') as pickle_file:
            test_rewards = pickle.load(pickle_file)
    except:
        test_rewards = []
    
    with open(path + "/" + filename + "_frame_idx", 'rb') as pickle_file:
        frame_idx = pickle.load(pickle_file)

    return (replay_buffer, rewards, test_rewards, frame_idx)

def test(render=False, random=0, test_sims=True):
    # Run the test initial positions of the environment
    episode_reward = []
    if test_sims: env.add_to_test(1)
    while True:
        state = env.reset()
        episode_reward.append(0)
        # Render into buffer. 
        for step in range(max_steps):
            if render:
                env.render()
            action = policy_net.get_action(state, random).detach()
            state, reward, done, info = env.step(action.numpy())
            episode_reward[-1]+=reward
            if done:
                print("\rTest case {:d}: Episode reward: {:f}".format(env.get_test(), episode_reward[-1]), end="\n")
                if test_sims: env.add_to_test(1)
                break
        if env.get_test()==0 and test_sims:
            break
    env.test = 0            
    if render:        
        env.close()

    return episode_reward

def norm_weight():
    max_values = np.zeros(state_dim, dtype=float)

    env.reset()
    #print("##################### CHANGE NORMWEIGHT BACK!!!! ########################")
    for i in range(10000):
        action = 2*np.random.random(action_dim)-1
        next_state, _, done, _ = env.step(action)

        max_values = np.maximum(np.absolute(next_state), max_values)

        if done:
            env.reset()

    norm_weight = 1/(max_values)
    norm_weight[norm_weight == inf] = 1

    return norm_weight

####################################################################
#                       START OF MAIN FUNCTION
#
####################################################################

TRAIN = 1 # 0 = start from scratch, 1 = continue from previous, 2 = test from previous
saving = 0 # 0 = not saving, 1 = saving
env_name = "KevinFallingHumanoid-v0"
env = NormalizedActions(gym.make(env_name))
if TRAIN == 0:
    now = datetime.datetime.now()
    directory_name = env_name + "_" + now.strftime("%m-%d-%H-%M") + " (exo, normal, 100Hz, old input, forward, LP)"
    if saving==1:
        os.makedirs("../saves/" + directory_name)
    else:
        print("No save file created")
else:
    directory_name = "KevinFallingHumanoid-v0_06-12-14-16 (exo, normal, 100Hz, old input, forward, LP)"

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]
hidden_dim = 512

if TRAIN==0:
    #print("NO WEIGHTS")
    #norm_weights = np.ones(state_dim)
    norm_weights = norm_weight()
else:
    norm_weights = np.ones(state_dim)

value_net        = ValueNetwork(state_dim, hidden_dim, norm_weights).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim, norm_weights).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, norm_weights).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, norm_weights).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, norm_weights).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 100000
replay_buffer = ReplayBuffer(replay_buffer_size)
temp_buffer = ReplayBuffer(replay_buffer_size)

max_frames  = 5000000
max_steps   = 1000
obs_frames  = 1000
frame_idx   = 0
rewards     = []
test_rewards= []
batch_size  = 128
alpha       = 1.0       # Relative weight of entropy
entropy_decay = 0.999   # Exponential decay of alpha

#plottime = 0

#Load function
if TRAIN != 0:
    replay_buffer, rewards, test_rewards, frame_idx = load()
    alpha = alpha*entropy_decay**(int(frame_idx/1000))

if TRAIN != 2:
    try:
        while frame_idx < max_frames:
            state = env.reset()
            '''
            for i in range(10000):
                action = 0*action_dim
                _, reward, _, _ = env.step(action)
                #print(reward)
                if i % 200==0:
                    env.reset()
                    #raise SystemExit(0)
                env.render()
            continue
            raise SystemExit(0)
            '''
            episode_reward = []
            
            for step in range(max_steps):
                if frame_idx > obs_frames:
                    action = policy_net.get_action(state).detach()
                    next_state, reward, done, _ = env.step(action.numpy())
                else:
                    action = 2*np.random.random(action_dim)-1
                    next_state, reward, done, _ = env.step(action)
                
                temp_buffer.push(state, action, reward, next_state, done)
                #replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward.append(reward)
                frame_idx += 1
                '''
                if plottime >= 600:
                    update(batch_size, alpha)
                else:
                    plottime+=1
                '''
                if len(replay_buffer) > batch_size:
                    update(batch_size, alpha)
                
                if frame_idx % 1000 == 0:
                    alpha*=entropy_decay

                if done:
                    break
            
            episode_reward = replay_buffer.insert(temp_buffer, episode_reward)

            average_episode_reward = sum(episode_reward)/step
            rewards.append(average_episode_reward)
            if len(rewards)>=25:
                average_reward = np.mean(rewards[-25:])
            else:
            	average_reward = np.mean(rewards)
            print("\rTotal T: {:d}  Reward: {:f} Avg frame Reward: {:f} Avg avg frame Reward: {:f}".format(frame_idx, sum(episode_reward), average_episode_reward, average_reward), end="\n\n")
            if len(rewards)%50 == 0:
                test_rewards.append(test())

    except KeyboardInterrupt:
        frame_idx = math.floor(frame_idx/200)*200

    if saving==1:
        print("Saving final model\n")
        save("end", "saves")
    else:
        print("Stopped without saving")
        raise SystemExit(0)
if test_rewards!=[]:
    plot(frame_idx, np.sum(test_rewards, 1)/9, "Average reward per episode over 9 tests", xlabel="Episodes")
else:
    plot(frame_idx, rewards, "Average reward per frame in episode", xlabel="Episodes")
test(True, 0, False)
