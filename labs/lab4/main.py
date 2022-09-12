# 1. Import Necessary Packages (e.g. `gym`, `numpy`):

import pygame
from pygame.locals import *
import sys, time, random
import numpy as np
import gym
import random
import pygame
from pynput import keyboard

"""
pygame.init()
env = gym.make("Taxi-v3", render_mode='human')
env.reset()
env.render()
"""

# 2. Instantiate the Environment and Agent

#env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)

action_size = env.action_space.n

state_size = env.observation_space.n

# 3. Set up the QTable:
learning_rate = 0.2
discount_rate = 0.9

q_table = np.zeros((state_size, action_size))
#q_table = np.random((state_size, action_size))


def update_q_table(reward, prev_state, action, state):
    q_table[prev_state, action] = q_table[prev_state, action] + learning_rate * \
                                   (reward + discount_rate * np.max(q_table[state]) - q_table[prev_state, action])

def get_best_action(state):
    return np.argmax(q_table[state])

"""
def get_best_action(state, action_space):
    best_action = [-1] * 4
    for a in action_space:
        best_action[a] = q_table[state][a]
    return np.argmax(best_action)

"""

# 4. The Q-Learning algorithm training
env.action_space.seed(42)
observation, info = env.reset(seed=42)
exploration = 0.25

# observation, info = env.reset(seed=42)

# Train
for i in range(0, 10000):
    last_observation = env.observation_space.sample()
    terminated = False

    while not terminated:
        explo = random.uniform(0, 1)

        if exploration < explo:
            action = get_best_action(last_observation)
        else:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        update_q_table(reward, last_observation, action, observation)

        last_observation = observation

        if terminated or truncated:
            observation, info = env.reset()

    #print("index:", i)

print(q_table)

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')

observation, info = env.reset(seed=42)

for i in range(0, 10):
    last_observation = env.observation_space.sample()
    terminated = False

    while not terminated:
        explo = random.uniform(0, 1)

        action = get_best_action(last_observation)

        observation, reward, terminated, truncated, info = env.step(action)

        # update_q_table(reward, last_observation, action, observation)

        last_observation = observation

        if terminated or truncated:
            observation, info = env.reset()

#render:




env.close()




#print(q_table)


"""
class action_value:
    def __init__(self):
        self.state = None
        self.action = None
        
    def get_string(self):
        str(self.state) + str(self.action)

class Q_Table:
    def __init__(self):
        self.learning_rate = 0.5
        self.discount_rate = 0.9
        self.q_table = {} # Dictionary that maps from state and action to value
        self.state_to_actions = {}

    def update(self, action_value):
        av = action_value.get_string()
        if av not in self.q_table:
            self.put(av, 0)

        self.q_table[av] += self.learning_rate *


    def get_q(self, action_value):
        av = action_value.get_string()
        if av in self.q_table:
            return self.q_table[av]
        else:
            return 0

    def max_a(self, action_value):
        max_a = -1000
        for a in action_value.


    def put(self, action_value, value):
        self.q_table[action_value] = value
"""