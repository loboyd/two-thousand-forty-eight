#!/usr/bin/env python3

import pickle
import random

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import Direction, Game, State

seed = 42
random.seed(seed)
torch.manual_seed(seed)

import pickle

# Define the neural network architecture
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x): return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)

    def save(self):
        with open('data.pickle', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls):
        with open('data.pickle', 'rb') as file:
            return pickle.load(file)


class Episode:
    def __init__(self, net, gamma=1):
        self.net = net
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        self.score = 0

    def run(self):
        game = Game(exp=True)
        ct = 0 # number of actions taken
        while game.state == State.ONGOING:
            # prepare input from board
            input_data = torch.tensor([[float(tile) for row in game.board for tile in row]])

            # run input through net generate policy distribution
            distribution = Categorical(self.net(input_data))

            # sample the distribution
            sample = distribution.sample()

            # write down the logprob
            self.log_probs.append(distribution.log_prob(sample))

            # convert action to direction/move
            direction = Direction(sample.item() + 1)

            # execute move
            score = game.score
            game.move(direction)
            ct += 1

            # write down the reward (change in score)
            self.rewards.append(game.score - score)

        # adjust rewards to represent future discounted return
        R = 0
        for i in range(len(self.rewards)-1, -1, -1):
            R = self.rewards[i] + self.gamma * R
            self.rewards[i] = R

        # compute total episode loss
        losses = [-log_prob * R for (log_prob, R) in zip(self.log_probs, self.rewards)]
        loss = torch.cat(losses).sum()

        # compute and accumulate gradient
        loss.backward()

        # write down final score
        self.score = game.score

        return ct

# set up net and optimizer
net = Agent()
optimizer = torch.optim.Adam(net.parameters())

#for epoch in range(20): # number of epochs
epoch = 0
while True:
    epoch += 1
    optimizer.zero_grad()

    ct = 0
    scores = []
    for _ in range(200): # number of episodes per epoch
        episode = Episode(net)
        ct += episode.run()
        scores.append(episode.score)

    for param in net.parameters():
        param.grad /= ct

    optimizer.step()

    net.save()
    print(f'epoch {epoch}: [{min(scores)}, {sum(scores) / len(scores)}, {max(scores)}]')

