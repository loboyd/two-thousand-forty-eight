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
        self.fc1 = nn.Linear(16, 512, bias=False)
        self.fc2 = nn.Linear(512, 32, bias=False)
        self.fc3 = nn.Linear(32, 32, bias=False)
        self.fc4 = nn.Linear(32, 4, bias=False)

        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.uniform_(self.fc4.weight, a=-0.01, b=0.01)

    def forward(self, x, mask, handle_symmetries=False):
        if handle_symmetries:
            x = Agent._symmetrify(x)

        x = x.view(-1, 16)

        # the actual math
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

    def _get_batch_distributions(self, states):
        boards = torch.tensor([state.board for state in states], dtype=torch.float32)
        masks = torch.tensor([state.get_move_mask() for state in states], dtype=torch.float32)

        return self.forward(boards, masks)

    def get_batch_moves(self, states, train=False):
        """Return a list of `Direction`s given a list of `Game`s."""
        distributions = self._get_batch_distributions(states)

        if train:
            move_indices = Categorical(distributions).sample().tolist()
        else:
            move_indices = torch.argmax(distributions, dim=1).tolist()

        return [Direction(move_index + 1) for move_index in move_indices]

    def get_move(self, game, train=False):
        """Returns the `Direction` given a `Game` if there is at least one legal move. Otherwise,
           returns `None`.."""
        # check move availability; return early if none available
        move_mask = game.get_move_mask()
        if move_mask == [False] * 4:
            return None

        # build tensors from `game.board` and `move_mask`
        board = torch.tensor(game.board, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(move_mask, dtype=torch.float32).unsqueeze(0)

        # run the network to compute probability distribution over action space
        distribution = self.forward(board, mask)

        # if training, sample policy distribution, otherwise, be greedy
        if train:
            move_index = Categorical(distribution).sample().item()
        else:
            move_index = torch.argmax(distribution, dim=1).item()

        # convert action to direction/move
        return Direction(move_index + 1)

    def update(self, optimizer, episodes):
        # prep data
        states = [state for ep in episodes for state in ep.states]
        actions = torch.tensor([action.value - 1 for ep in episodes for action in ep.actions])
        rewards = torch.tensor([reward for ep in episodes for reward in ep.rewards])
        scores = [ep.score for ep in episodes] # only used for progress reporting

        # normalize rewards (sigh... i guess)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # do a forward pass (and reconstruct the sampling using the actions)
        distributions = self._get_batch_distributions(states)
        as_sampled = distributions[torch.arange(actions.size(0)), actions]

        # compute loss
        action_log_probs = torch.log(as_sampled)
        loss = (-action_log_probs * rewards).sum()

        # zero grad and do a backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report on progress
        grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in self.parameters()]), p=2)
        print(f'grad norm: {grad_norm}')
        min_score = min(scores)
        avg_score = sum(scores)/len(scores)
        max_score = max(scores)
        print(f'[{min_score:7.2f}, {avg_score:7.2f}, {max_score:7.2f}]')
        print()

    def save(self):
        with open('data.pickle', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls):
        with open('data.pickle', 'rb') as file:
            return pickle.load(file)

