import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game import Direction, Game

def set_seed(seed): torch.manual_seed(seed)

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.convv = nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)
        self.convh = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
        self.fc1 = nn.Linear(16+12+12, 512, bias=False)
        self.fc2 = nn.Linear(512, 32, bias=False)
        self.fc3 = nn.Linear(32, 32, bias=False)
        self.fc4 = nn.Linear(32, 4, bias=False)

        nn.init.xavier_uniform_(self.convv.weight)
        nn.init.xavier_uniform_(self.convh.weight)
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.uniform_(self.fc4.weight, a=-0.01, b=0.01)

    def conv(self, x):
        """Perform all the convs, concat the outputs, ReLU-activate"""
        n, _, _ = x.shape
        v = self.convv(x.unsqueeze(1)).view(n, 12) # this conv has shape (n, 1, 3, 4)
        h = self.convh(x.unsqueeze(1)).view(n, 12) # this conv has shape (n, 1, 4, 3)
        x = x.view(n, 16)
        x = torch.cat((x, v, h), dim=1)
        return F.relu(x)

    def forward(self, x, mask, handle_symmetries=True):
        if handle_symmetries:
            x = Agent._symmetrify(x)

        n, _, _ = x.shape

        # the actual math
        x = self.conv(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        if handle_symmetries:
            x = Agent._unsymmetrify(x)

        # enforce mask
        x[mask == 0] = float('-inf')

        x = F.softmax(x, dim=1)
        return x

    def _symmetrify(x):
        """Augment the given batched input with dihedral-symmetric transformations."""
        n, nr, nc = x.shape

        # generate board symmetries
        x_r = torch.rot90(x, 1, (1, 2))
        x_rr = torch.rot90(x, 2, (1, 2))
        x_rrr = torch.rot90(x, 3, (1, 2))
        x_f = torch.flip(x, [2])
        x_rf = torch.flip(x_r, [2])
        x_rrf = torch.flip(x_rr, [2])
        x_rrrf = torch.flip(x_rrr, [2])

        x = torch.stack([x, x_r, x_rr, x_rrr, x_f, x_rf, x_rrf, x_rrrf])

        return x.view(8*n, nr, nc)

    def _unsymmetrify(x):
        # transforms on the action space: rotate counterclockwise and flip about vertical axis
        def r(index=torch.arange(4)): return index[torch.tensor([1, 2, 3, 0])]
        def f(index=torch.arange(4)): return index[torch.tensor([0, 3, 2, 1])]

        n, _ = x.shape

        # split symmetries into their own 8-element dimension
        x = x.view(8, n//8, 4)

        # untransform the action distributions
        x[1, :, :] = x[1, :, r(r(r()))] # 3*r = -1*r
        x[2, :, :] = x[2, :, r(r())]
        x[3, :, :] = x[3, :, r()] # 1*r = -3*r
        x[4, :, :] = x[4, :, f()]
        x[5, :, :] = x[5, :, r(r(r(f())))] # 3*r = -1*r
        x[6, :, :] = x[6, :, r(r(f()))] # 3*r = -1*r
        x[7, :, :] = x[7, :, r(f())]

        # sum across symmetries
        return torch.sum(x, dim=0)

    def get_move(self, game, train=False):
        # prepare input from board
        move_mask = game.get_move_mask()
        if move_mask == [False] * 4:
            return None

        # build tensors from `game.board` and `move_mask`
        board = torch.tensor(game.board, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(move_mask, dtype=torch.float32).unsqueeze(0)

        # run the network
        output = self.forward(board, mask)

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

