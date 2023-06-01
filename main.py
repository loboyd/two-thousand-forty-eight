#!/usr/bin/env python3

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

# Define the neural network architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(16, 100)  # Fully connected layer from input to hidden
        self.relu = nn.ReLU()  # Activation function (ReLU)
        self.fc2 = nn.Linear(100, 4)  # Fully connected layer from hidden to output

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)  # Apply softmax activation to the output
        return out


class Replay:
    def __init__(self, net):
        #self.seed = random.randint(0, 1023)
        self.net = net
        self.states = [] # as input-ready tensors
        self.actions = []
        self.log_probs = []
        self.rewards = []

    def play(self, display=False):
        game = Game()
        if display: print(game)
        while game.state == State.ONGOING:
            # format input and write down state
            flattened_board = [float(item) for sublist in game.board for item in sublist]
            input_data = torch.tensor([flattened_board])
            self.states += [input_data]

            # pass the board state into the network
            output = net(input_data)

            # sample the output; compute and write down log_prob
            distribution = Categorical(output)
            sample = distribution.sample()
            self.log_probs += distribution.log_prob(sample)

            # determine the action based on the sample, write it down
            action = Direction(sample.item() + 1)
            self.actions += [action]
            if display: print(action)
            game.move(action)

            # write down the current score
            self.rewards += [game.score]

            if display:
                for _ in range(100): print()
                print(game)
                print(game.score)

        self._adjust_rewards(game.score)

    def _adjust_rewards(self, final_score):
        # adjust to represent all future reward (todo: maybe discount)
        for r in range(len(self.rewards)):
            self.rewards[r] = final_score - self.rewards[r]

    def grad(self):
        """Accumulates the gradient computed according to this `Replay`. Returns the number of
           components to the gradient (caller should divide by this number)"""
        n = len(replay.states)
        for i in range(n):
            state = replay.states[i]
            action = replay.actions[i]
            log_prob = replay.log_probs[i]
            reward = replay.rewards[i]

            loss = -log_prob * reward
            loss.backward() # accumulate the gradient (to be normalized later)
        return n

net = SimpleNet()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# run a single game trajectory
replay = Replay(net)
replay.play()
n = replay.grad()
for param in net.parameters():
    param.grad /= n
    print(param.grad)

assert(len(replay.states) == len(replay.actions) == len(replay.log_probs) == len(replay.rewards))

"""
n = len(replay.states)
for i in range(n):
    state = replay.states[i]
    action = replay.actions[i]
    log_prob = replay.log_probs[i]
    reward = replay.rewards[i]

    #print(f'state: {state}')
    #print(f'action: {action}')
    #print(f'log_prob: {log_prob}')
    #print(f'reward: {reward}')
    #print(f'log_prob * reward: {log_prob * reward}')
    #print()

    loss = -log_prob * reward
    loss.backward() # accumulate the gradient (to be normalized later)

for param in net.parameters():
    param.grad /= n
    print(param.grad)
"""

#optimizer.step()

