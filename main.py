#!/usr/bin/env python3

LOAD = False
SAVE = True
BATCH_SIZE=512

import random
import time

import torch

from agent import Agent, set_seed
from episode import Episode

seed = 40
random.seed(seed)
set_seed(seed)

def update(net, optimizer, episodes, start_time=None):
    if not start_time:
        start_time = time.time()

    ## this is the BATCH training update logic #####################################################

    # make biiiiig lists
    states = [state[0] for ep in episodes for state in ep.states]
    masks = [state[1] for ep in episodes for state in ep.states]
    actions = [action for ep in episodes for action in ep.actions]
    rewards = [float(reward) for ep in episodes for reward in ep.rewards]
    scores = [ep.score for ep in episodes] # used only for progress printing

    # pack shit into tensors
    states = torch.stack(states)
    masks = torch.stack(masks)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)

    # normalize rewards (sigh... i guess)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    action_probs = net.forward(states, masks)
    action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze())
    loss = (-action_log_probs * rewards).sum()

    # perform backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ################################################################################################

    if SAVE: net.save()

    min_score = min(scores)
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    dt = time.time() - start_time

    # todo: remove this
    grad_norm = 0
    for param in net.parameters():
        grad_norm += (param.grad ** 2).sum().item()
    grad_norm = grad_norm ** 0.5
    print(f'grad norm: {grad_norm}')
    print(f'[{min_score:7.2f}, {avg_score:7.2f}, {max_score:7.2f}], {dt:4.2f}s')


# NOTE: Look into RMSprop optimization
# set up net and optimizer
net = Agent.load() if LOAD else Agent()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
#lr_decay = lambda epoch: 0.1 ** (epoch // 16)
lr_decay = lambda epoch: max(0.0001, 0.9 ** (epoch // 4)) # decay by 90% every 32 epochs, plateau
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay)

while True:
    start_time = time.time()

    # todo: now this can be parallelized
    episodes = [Episode(net) for _ in range(BATCH_SIZE)]
    for ep in episodes: ep.run()
    update(net, optimizer, episodes, start_time)
    scheduler.step()
    print(f'lr: {optimizer.param_groups[0]["lr"]}')
    print()

