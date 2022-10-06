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
    def __init__(self, observation_size, action_size):
        super(Model, self).__init__()
        self.dense1 = nn.Linear(observation_size, 100)
        torch.nn.init.xavier_uniform_(self.dense1.weight)
        self.dense2 = nn.Linear(100, 100)
        torch.nn.init.xavier_uniform_(self.dense2.weight)
        self.dense3 = nn.Linear(100, action_size)
        torch.nn.init.xavier_uniform_(self.dense3.weight)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        return x

    def predict(self, x):
        x = torch.tensor(x)
        x = self.forward(x)
        return torch.argmax(x) 

class Agent:
    def __init__(self, observation_size, action_size):
        self.observation_size=observation_size
        self.action_size = action_size
        self.criterion = nn.MSELoss()
        self.model = Model(observation_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.N = 2000
        self.explore_rate = 0.15
        self.explore_decay = 0.99
        self.explore_min = 0.0
        self.discount_rate = 0.9
        #self.memory = Queue.queue(self.N)
        self.memory = deque([], maxlen=self.N)
        # self.memory = torch.tensor(np.array.zeros(self.N, 4)) # memory that stores N most new transitions
        # good place to store hyperparameters as attributes

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state, use_random=True):
        if random.random() < self.explore_rate and use_random:
            return random.randint(0, 1)

        action = self.model.predict(state)
        action = int(action)
        return action

    def replay(self, batch_size):

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
        pred = self.model.forward(s)[a]
        loss = self.criterion(pred, v)
        loss.backward()


def train(env, agent, episodes=10000, batch_size=64):  # train for many games
    plt.ion()
    plt.show()
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.title('Cart pole')
    # x axis values
    x = []
    # corresponding y axis values
    y = []

    for e in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        total_r = 0
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
        #print("score: ", total_r)
        #print("explore: ", agent.explore_rate)

        # plotting
        x.append(e)
        y.append(total_r)

        if e % 10 == 0:
            plt.plot(x, y)

            plt.draw()
            plt.pause(0.1)
    env.close()


env = gym.make('CartPole-v1' , render_mode='human')
agent = Agent(env.observation_space.shape[0], env.action_space.n)
#agent.model.state_dict = torch.load('model3.pth')
train(env, agent)
torch.save(agent.model.state_dict(), 'model3.pth')


#torch.load('model2.pth')
#agent = Agent(env.observation_space.shape[0], env.action_space.n)

agent.model.load_state_dict(torch.load('model2.pth'))

#print(agent.model.state_dict)

env = gym.make('CartPole-v1', render_mode='human')

for _ in tqdm(range(100)):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.act(state, use_random=False)
        # print(action)
        state, reward, done, _, _ = env.step(action)

env.close()
