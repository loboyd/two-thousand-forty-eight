#!/usr/bin/env python3

LOAD = True
SAVE = True

import random
import time

import torch

from agent import Agent, Episode, set_seed

seed = 42
random.seed(seed)
set_seed(seed)

# set up net and optimizer
net = Agent.load() if LOAD else Agent()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)

#for epoch in range(20): # number of epochs
epoch = 0
while True:
    t = time.time()
    epoch += 1
    optimizer.zero_grad()

    ct = 0
    scores = []
    for _ in range(2000): # number of episodes per epoch
        episode = Episode(net)
        ct += episode.run()
        scores.append(episode.score)

    for param in net.parameters():
        param.grad /= ct

    optimizer.step()

    if SAVE: net.save()
    print(f'epoch {epoch}: [{min(scores):7.2f}, {(sum(scores) / len(scores)):7.2f}, {max(scores):7.2f}], {(time.time() - t):4.2f}s')

