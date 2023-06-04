#!/usr/bin/env python3

import pickle
import random

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import Direction, Game, State

#seed = 41
#random.seed(seed)
#torch.manual_seed(seed)

import pickle

def save(data):
    with open('data.pickle', 'wb') as file:
        pickle.dump(data, file)

def load():
    with open('data.pickle', 'rb') as file:
        data = pickle.load(file)

    return data

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
        self.game = Game(exp=False)
        self.states = [] # as input-ready tensors
        self.actions = []
        self.log_probs = []
        self.rewards = []
        # todo: store `n` number of actions here, computed during `self.play()` instead of `self.grad()`

    def play(self, display=False):
        if display: print(self.game)
        while self.game.state == State.ONGOING:
            # format input and write down state
            flattened_board = [float(item) for sublist in self.game.board for item in sublist]
            input_data = torch.tensor([flattened_board])
            self.states.append(input_data)

            # pass the board state into the network
            output = net(input_data)

            # sample the output; compute and write down log_prob
            distribution = Categorical(output)
            sample = distribution.sample()
            self.log_probs.append(distribution.log_prob(sample))

            # determine the action based on the sample, write it down
            action = Direction(sample.item() + 1)
            self.actions.append(action)
            if display: print(action)
            self.game.move(action)

            # write down the current score
            self.rewards.append(self.game.score)

            if display:
                for _ in range(100): print()
                print(self.game)
                print(self.game.score)

        self._adjust_rewards(self.game.score)

    def _adjust_rewards(self, final_score):
        # adjust to represent all future reward (todo: maybe discount)
        for r in range(len(self.rewards)):
            self.rewards[r] = final_score - self.rewards[r]

    def grad(self):
        """Accumulates the gradient computed according to this `Replay`. Returns the number of
           components to the gradient (caller should divide by this number)"""
        n = len(self.states)
        for i in range(n):
            state = self.states[i]
            action = self.actions[i]
            log_prob = self.log_probs[i]
            reward = self.rewards[i]

            loss = -log_prob * reward
            loss.backward() # accumulate the gradient (to be normalized later)
        return n


class Batch:
    def __init__(self, net, optimizer=None, batch_size=100):
        """Handle a batch of Replays"""
        self.net = net
        self.optimizer = optimizer if optimizer is not None else optim.Adam(net.parameters(), lr=0.003)
        self.batch_size = batch_size
        self.replays = []
        self.total_action_count = 0

    def update(self):
        self.run()
        scores = [replay.game.score for replay in self.replays]
        min_score = min(scores)
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        print(f'[{min_score:>4}, {mean_score:>8}, {max_score:>4}]')
        self.optimizer.step()

        # clean up
        self.net.zero_grad()
        self.replays.clear()
        self.total_action_count = 0

    def run(self, show_progress=False):
        """Run `self.batch_size` different games (`Replay`s), and compute gradients"""
        for r in range(self.batch_size):
            if show_progress and r % 10 == 0: print(f'running batch {r}')
            replay = Replay(net)
            replay.play()
            n_actions = replay.grad()
            self.total_action_count += n_actions
            self.replays.append(replay)
        for param in self.net.parameters():
            param.grad /= self.total_action_count

#net = load()
net = SimpleNet()

batch = Batch(net, batch_size=20)
while True:
    batch.update()
    save(net)

#save(net)

