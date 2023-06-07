import pickle
import random

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import Direction, Game, State

def set_seed(seed): torch.manual_seed(seed)

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(20, 512)
        self.fc2 = nn.Linear(512, 4)

    # todo: there is an extra input dimension that flows all the way to the output. get rid of it.
    def forward(self, x, mask):
        x = torch.cat((x, mask), dim=2)
        out = self.fc2(F.relu(self.fc1(x)))
        out = F.softmax(out * mask, dim=1)
        return out.squeeze(1)

    def play_move(self, game):
        # prepare input from board
        input_data = torch.tensor([[float(tile) for row in game.board for tile in row]]).unsqueeze(0)
        mask = torch.tensor([[float(x) for x in game.get_available_moves()]]).unsqueeze(0)

        # run input through net to generate policy distribution
        distribution = Categorical(self.forward(input_data, mask))

        # sample the distribution
        sample = distribution.sample()

        # convert action to direction/move and execute
        game.move(Direction(sample.item() + 1))

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

    def run(self):
        game = Game(exp=True)
        ct = 0 # number of actions taken
        while game.state == State.ONGOING:
            # prepare input from board
            # todo: could probably handle the squeeze stuff better
            input_data = torch.tensor([[float(tile) for row in game.board for tile in row]]).unsqueeze(0)
            mask = torch.tensor([[float(x) for x in game.get_available_moves()]]).unsqueeze(0)
            self.states.append((input_data.squeeze(0), mask.squeeze(0)))

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

