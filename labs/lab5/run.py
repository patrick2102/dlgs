from pickletools import optimize
from queue import Queue
from re import A, S
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque  # for memory
from tqdm import tqdm          # for progress bar


env = gym.make('CartPole-v1', render_mode='human')
"""
for _ in tqdm(range(10)):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)

"""
env.close()



#Model:
class Model(nn.Module):
    def __init__(self, observation_size, action_size):
        super(Model, self).__init__()
        #self.conv1 = nn.Conv2d(observation_size, 32, 3, 1)
        self.dense1 = nn.Linear(observation_size, 500)
        self.dense2 = nn.Linear(500, action_size)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

    def predict(self, x):              # actually make a prediction
        x = torch.tensor(x)
        x = self.forward(x)            # send x through neural net
        return torch.argmax(x, dim=1)  # predict most likely thing

class Agent:
    def __init__(self, observation_size, action_size):
        self.observation_size=observation_size
        self.action_size = action_size
        self.criterion = nn.MSELoss()
        self.model = Model(observation_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.N = 2000
        #self.memory = Queue.queue(self.N)
        self.memory = deque([], maxlen=self.N)
        # self.memory = torch.tensor(np.array.zeros(self.N, 4)) # memory that stores N most new transitions
        # good place to store hyperparameters as attributes

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        state = torch.tensor(state)
        action = self.model(state)
        action = torch.argmax(action)
        action = int(action)

        # print("action: ", action)

        return action

    def replay(self, batch_size):
        while len(self.memory) > 0:
            self.train()
        # update model based on replay memory
        # you might want to make a self.train() helper method

    def train(self):
        s, a, r, s1, d = self.memory.pop()
        s = torch.tensor(s)
        r = torch.tensor(r)
        pred = self.model(s) 
        loss = self.criterion(pred, r)
        loss.backward()
        self.optimizer.step()


def train(env, agent, episodes=1000, batch_size=64):  # train for many games
    for _ in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        while not done:
            # 1. make a move in game.
            action = agent.act(state)
            #print(action)
            next_state, reward, done, _, _ = env.step(action)

            # 2. have the agent remember stuff.
            agent.remember(state, action, reward, next_state, done)

            # 3. update state
            state = next_state

            # 4. if we have enough experiences in out memory, learn from a batch with replay.
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)
    env.close()


env = gym.make('CartPole-v1')  # , render_mode='human')
agent = Agent(env.observation_space.shape[0], env.action_space.n)
train(env, agent)
torch.save(agent.model.state_dict(), 'model.pth')