#!/usr/bin/env python3

import random

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

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


# todo: probably put this in a class or something
def sample_action(output):
    distribution = Categorical(output)
    action = distribution.sample()
    return action.item()


class Replay:
    def __init__(self, net):
        #self.seed = random.randint(0, 1023)
        self.net = net
        self.states = [] # as input-ready tensors
        self.actions = []
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

            # sample the output to determine which action to play, write down the action
            action = Direction(sample_action(output) + 1)
            self.actions += [action]
            if display: print(action)
            game.move(action)

            # write down the current score
            self.rewards += [game.score]

            if display:
                for _ in range(100): print()
                print(game)
                print(game.score)


# Create an instance of the network
net = SimpleNet()
replay = Replay(net)
replay.play()
for (state, action, reward) in zip(replay.states, replay.actions, replay.rewards):
    print(f'state: {state}')
    print(f'action: {action}')
    print(f'reward: {reward}')
    print()

