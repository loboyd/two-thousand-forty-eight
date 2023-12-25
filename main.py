#!/usr/bin/env python3

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from agent import Agent
from episode import Batch

BATCH_SIZE = 2**7
LEARNING_RATE = 0.0001

mx.set_default_device(mx.gpu)

agent = Agent()

def evaluate(batch):
    scores = list(sorted([ep.score for ep in batch.episodes]))
    print(f'{scores[0]:4.0f}, {scores[len(scores)//2]:4.0f}, {scores[-1]:4.0f}')

def _loss_fn(model, boards, masks, actions, rewards):
    distributions = model.forward(boards, masks)
    as_sampled = distributions[mx.arange(actions.shape[0]), actions]
    action_log_probs = mx.log(as_sampled)
    return (-action_log_probs * rewards).sum()

#---------------------------------------------------------------------------------------------------
def train(agent, optimizer):
    batch = Batch(agent, batch_size=BATCH_SIZE, gamma=0.96)
    batch.run()

    states = [state for ep in batch.episodes for state in ep.states]
    actions = mx.array([action.value - 1 for ep in batch.episodes for action in ep.actions])
    rewards = mx.array([reward for ep in batch.episodes for reward in ep.rewards])
    scores = [ep.score for ep in batch.episodes] # only used for progress reporting

    # normalize rewards (sigh... i guess)
    std = (((rewards - (m := rewards.mean()))**2).sum() / rewards.shape[0]).sqrt()
    rewards = (rewards - rewards.mean()) / (std + 1e-5)

    boards = mx.array([state.board for state in states])
    masks = mx.array([state.get_move_mask() for state in states])

    loss_and_grad_fn = nn.value_and_grad(agent, _loss_fn)
    loss, grads = loss_and_grad_fn(agent, boards, masks, actions, rewards)

    print(f'loss: {loss.item():.2f}')

    #for layer in grads['layers']:
    #    print(layer['weight'])

    for i, layer in enumerate(grads['layers']):
        agent.layers[i].weight -= LEARNING_RATE * layer['weight']
    mx.eval(agent.parameters())

    # using provided optimizer
    #mx.eval(agent.parameters(), optimizer.state)

    evaluate(batch)

optimizer = optim.SGD(learning_rate=LEARNING_RATE)
t = time.time()
while True:
    train(agent, optimizer)
    print(f'that took {time.time() - t:3.2f} seconds')
    print()

    t = time.time()

