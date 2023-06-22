import pickle
import random

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import Direction, Game

def set_seed(seed): torch.manual_seed(seed)

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(20, 512, bias=False)
        self.fc2 = nn.Linear(512, 4, bias=False)

        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.uniform_(self.fc1.weight, a=-0.01, b=0.01)

    def forward(self, x, mask):
        x = torch.cat((x, mask), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x[mask == 0] = float('-inf')
        x = F.softmax(x, dim=1)
        return x

    def play_move(self, game, train=False):
        """Execute a move according to the policy. If `train == false`, play greedily. Returns
           boolean describing whether or not a move was successful, i.e, if there were any moves
           available."""
        move = self.get_move(game, train=train)
        if move is None:
            return False
        game.move(move)
        return True

    def get_move(self, game, train=False):
        # prepare input from board
        move_mask = game.get_move_mask()
        if move_mask == [False] * 4:
            return None

        input_data = torch.tensor([float(tile) for row in game.board for tile in row]).unsqueeze(0)
        mask = torch.tensor([float(x) for x in move_mask]).unsqueeze(0)

        # run the network
        output = self.forward(input_data, mask)

        # if training, sample policy distribution, otherwise, be greedy
        if train:
            move_index = Categorical(output).sample().item()
        else:
            move_index = torch.argmax(output, dim=1).item()

        # convert action to direction/move
        return Direction(move_index + 1)

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
        self.states = []
        self.actions = []
        self.rewards = []
        self.score = 0

    # todo: possibly refactor this to take advantage of the `Agent.get_move()` method (since right
    #       now that code is largely a duplication of this code.
    def run(self):
        game = Game(exp=True)
        ct = 0 # number of actions taken
        while True:
            mask = game.get_move_mask()
            if mask == [False, False, False, False]:
                break

            # prepare input from board
            input_data = torch.tensor([float(tile) for row in game.board for tile in row])
            mask = torch.tensor([float(x) for x in mask])
            self.states.append((input_data, mask))

            # add a dimension for batching
            input_data = input_data.unsqueeze(0)
            mask = mask.unsqueeze(0)

            # run input through net generate policy distribution
            distribution = Categorical(self.net(input_data, mask))

            # sample the distribution
            sample = distribution.sample()

            # convert action to direction/move
            sample_item = sample.item()
            direction = Direction(sample_item + 1)

            # execute move
            score = game.score
            game.move(direction)
            self.actions.append(sample_item)

            # write down the reward (change in score)
            self.rewards.append(game.score - score)

        self._adjust_rewards()

        self.score = game.score

    def _adjust_rewards(self):
        R = 0
        for i in range(len(self.rewards)-1, -1, -1):
            R = self.rewards[i] + self.gamma * R
            self.rewards[i] = R

