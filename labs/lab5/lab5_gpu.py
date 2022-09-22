from pickletools import optimize
from queue import Queue
from re import A, S

#from cv2 import sqrt
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque  # for memory
from tqdm import tqdm          # for progress bar


#Model:
class Model(nn.Module):
    def __init__(self, observation_size, action_size, device="cpu"):
        super(Model, self).__init__()
        self.dense1 = nn.Linear(observation_size, 100)
        torch.nn.init.xavier_uniform_(self.dense1.weight)
        self.dense2 = nn.Linear(100, 100)
        torch.nn.init.xavier_uniform_(self.dense2.weight)
        self.dense3 = nn.Linear(100, action_size)
        torch.nn.init.xavier_uniform_(self.dense3.weight)

        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        return x

    def predict(self, x):              # actually make a prediction
        x = torch.tensor(x)
        x = self.forward(x)            # send x through neural net
        #return torch.argmax(x)  # predict most likely thing
        return torch.argmax(x)  # predict predictmost likely thing

class Agent:
    def __init__(self, observation_size, action_size, device="cpu"):
        self.observation_size=observation_size
        self.action_size = action_size
        self.criterion = nn.MSELoss()
        self.model = Model(observation_size, action_size, device)
        self.model.to(device)
        if device == "cuda:0":
            print("running with GPU")
        else:
            print("running with CPU")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.N = 2000
        self.explore_rate = 0.15
        self.explore_decay = 0.99
        self.explore_min = 0.0
        self.discount_rate = 0.9
        #self.memory = Queue.queue(self.N)
        #self.memory = deque([], maxlen=self.N)
        self.memory = deque([], maxlen=self.N)
        self.device = device
        # self.memory = torch.tensor(np.array.zeros(self.N, 4)) # memory that stores N most new transitions
        # good place to store hyperparameters as attributes

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state, use_random=True):
        if random.random() < self.explore_rate and use_random:
            return random.randint(0, 1)

        action = self.model.predict(state)
        #action = torch.argmax(action)
        action = int(action)

        return action

    def replay(self, batch_size):

        total_loss = 0
        minibatch = random.sample(self.memory, batch_size)
        self.optimizer.zero_grad()

        for i in range(batch_size):
            self.train(minibatch[i])

        self.optimizer.step()

    def train(self, sample):
        s, a, r, s1, d = sample
        s = torch.tensor(s)
        s1 = torch.tensor(s1)
        r = torch.tensor(r)

        if not d:
            v = r + self.discount_rate * float(torch.max(self.model.forward(s1)))
        else:
            v = r
        s = s.to(self.device)
        v = v.to(self.device)
        pred = self.model.forward(s)[a]
        loss = self.criterion(pred, v)
        loss.backward()


def train(env, agent, episodes=1000, batch_size=64):  # train for many games

    for e in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        total_r = 0
        total_loss = 0
        iter = 0
        while not done:
            # 1. make a move in game.
            
            action = agent.act(state)

            #print(action)
            next_state, reward, done, _, _ = env.step(action)
 
            total_r += reward

            # 2. have the agent remember stuff.
            #agent.remember(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)

            # 3. update state
            state = next_state

            # 4. if we have enough experiences in out memory, learn from a batch with replay.
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)
            iter += 1
        
    env.close()


env = gym.make('CartPole-v1', render_mode='human')  # , render_mode='human')


if torch.cuda.is_available():
    agent = Agent(env.observation_space.shape[0], env.action_space.n, device="cuda:0")
else:
    agent = Agent(env.observation_space.shape[0], env.action_space.n)

train(env, agent)
