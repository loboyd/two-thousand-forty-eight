#!/usr/bin/env python3

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from game import Direction, Game, State

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


def sample_action(output):
    distribution = Categorical(output)
    action = distribution.sample()
    return action.item()

# Create an instance of the network
net = SimpleNet()

game = Game()
print(game)
while game.state == State.ONGOING:
    # pass the board state into the network
    flattened_board = [float(item) for sublist in game.board for item in sublist]
    input_data = torch.tensor([flattened_board])
    output = net(input_data)

    # get action to play
    action = Direction(sample_action(output) + 1)
    print(action)
    game.move(action)

    for _ in range(100): print()
    print(game)

