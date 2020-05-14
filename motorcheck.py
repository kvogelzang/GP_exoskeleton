import gym
import math
import numpy as np

from IPython.display import clear_output
import matplotlib.pyplot as plt

def plot(frame_idx, rewards, ylabel="", subplot=None, legend=None):
    clear_output(True)
    plt.figure(figsize=(10,5))
    if subplot:
        plt.subplot(subplot)
    #plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(np.arange(0, len(rewards)*framerate, framerate), rewards, marker='.', ms=1)
    plt.ylabel(ylabel)
    plt.xlabel("time [s]")
    plt.title("Frequency test at f={:d}Hz".format(freq))
    if legend:
        plt.legend(legend)
    plt.show()

env = gym.make("InvertedPendulum-v2")

freq = 30
framerate = 0.002
max_steps = int(30/(freq*framerate))
states = []
min_value = 0
max_value = 0

env.reset()
for step in range(max_steps):
    action = math.cos(step*framerate*freq)
    next_state, _, _, _ = env.step(action)
    states.append(next_state[2:4])

    if (next_state[2] < min_value):
        min_value = next_state[2]

    if next_state[2] > max_value:
        max_value = next_state[2]


print(max_value-min_value)
plot(max_steps, states, ylabel="amplitude", legend=["applied force", "applied control signal"])