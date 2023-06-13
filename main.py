#!/usr/bin/env python3

LOAD = False
SAVE = False
BATCH_SIZE=2000

import random
import time

import torch

from agent import Agent, Episode, set_seed

seed = 42
random.seed(seed)
set_seed(seed)

def update(net, optimizer, episodes):
    t = time.time()

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

    dt = time.time() - t
    min_score = min(scores)
    med_score = scores[len(scores)//2]
    max_score = max(scores)
    print(f'[{min_score:7.2f}, {med_score:7.2f}, {max_score:7.2f}], {dt:4.2f}s')


# set up net and optimizer
net = Agent.load() if LOAD else Agent()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)

while True:
    # todo: now this can be parallelized
    episodes = [Episode(net) for _ in range(BATCH_SIZE)]
    for ep in episodes: ep.run()
    update(net, optimizer, episodes)

