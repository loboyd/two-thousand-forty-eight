import torch
from torch.distributions import Categorical

from game import Direction, Game

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
            board = torch.tensor(game.board, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32)
            self.states.append((board, mask))

            # add a dimension for batching
            board = board.unsqueeze(0)
            mask = mask.unsqueeze(0)

            # run input through net generate policy distribution
            distribution = Categorical(self.net(board, mask))

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

