import pickle
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from game import Direction, Game

#def set_seed(seed): torch.manual_seed(seed)

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        #sizes = [16, 512, 32, 32, 4]
        sizes = [16, 4096, 4096, 4096, 32, 4]
        self.layers = [nn.Linear(i, o, bias=False) for i, o in zip(sizes[:-1], sizes[1:])]

    def forward(self, x, mask, handle_symmetries=False):
        x = x.reshape(-1, 16)
        for layer in self.layers[:-1]:
            x = mx.maximum(layer(x), 0.0)
        x = self.layers[-1](x)
        mx.where(mask == 0, float('-inf'), mask)
        x = mx.softmax(x, axis=1)
        return x

    def _get_batch_distributions(self, states):
        boards = mx.array([state.board for state in states])
        masks = mx.array([state.get_move_mask() for state in states])

        return self.forward(boards, masks)

    def get_batch_moves(self, states, train=False):
        """Return a list of `Direction`s given a list of `Game`s."""
        pred = self._get_batch_distributions(states)

        move_indices = (mx.random.categorical(pred) if train else mx.argmax(pred, dim=1)).tolist()

        return [Direction(move_index + 1) for move_index in move_indices]

    def get_move(self, game, train=False):
        """Returns the `Direction` given a `Game` if there is at least one legal move. Otherwise,
           returns `None`.."""
        # check move availability; return early if none available
        move_mask = game.get_move_mask()
        if move_mask == [False] * 4:
            return None

        # build tensors from `game.board` and `move_mask`
        board = mx.array(game.board).reshape(1, 4, 4)
        mask = mx.array(move_mask).reshape(1, 4)

        # run the network to compute probability distribution over action space
        pred = self.forward(board, mask)

        # if training, sample policy distribution, otherwise, be greedy
        move_index = (mx.random.categorical(pred) if train else mx.argmax(pred, axis=1)).item()

        # convert action to direction/move
        return Direction(move_index + 1)

    #def update(self, optimizer, episodes):
    #    # prep data
    #    states = [state for ep in episodes for state in ep.states]
    #    actions = mx.array([action.value - 1 for ep in episodes for action in ep.actions])
    #    rewards = mx.array([reward for ep in episodes for reward in ep.rewards])
    #    scores = [ep.score for ep in episodes] # only used for progress reporting

    #    # normalize rewards (sigh... i guess)
    #    std = (((rewards - (m := rewards.mean()))**2).sum() / rewards.shape[0]).sqrt()
    #    rewards = (rewards - rewards.mean()) / (std + 1e-5)

    #    boards = mx.array([state.board for state in states])
    #    masks = mx.array([state.get_move_mask() for state in states])

    #    def _loss_fn(model):
    #        distributions = model.forward(boards, masks)
    #        as_sampled = distributions[mx.arange(actions.shape[0]), actions]

    #        # compute loss
    #        action_log_probs = mx.log(as_sampled)
    #        return (-action_log_probs * rewards).sum()

    #    loss_and_grad_fn = nn.value_and_grad(self, _loss_fn)
    #    loss, grad = loss_and_grad_fn(self)
    #    print('created loss, grad')
    #    #print(grad.shape)

        #print(grad(episodes))

#        # zero grad and do a backward pass
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        # report on progress
#        grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in self.parameters()]), p=2)
#        print(f'grad norm: {grad_norm}')
#        min_score = min(scores)
#        avg_score = sum(scores)/len(scores)
#        max_score = max(scores)
#        print(f'[{min_score:7.2f}, {avg_score:7.2f}, {max_score:7.2f}]')
#        print()
#
#    def save(self):
#        with open('data.pickle', 'wb') as file:
#            pickle.dump(self, file)
#
#    @classmethod
#    def load(cls):
#        with open('data.pickle', 'rb') as file:
#            return pickle.load(file)
#
