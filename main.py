#!/usr/bin/env python3

import random
import time

import torch

from agent import Agent, Episode, set_seed

seed = 42
random.seed(seed)
set_seed(seed)

# set up net and optimizer
net = Agent.load()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)

#for epoch in range(20): # number of epochs
epoch = 0
while True:
    t = time.time()
    epoch += 1
    optimizer.zero_grad()

    ct = 0
    scores = []
    for _ in range(200): # number of episodes per epoch
        episode = Episode(net)
        ct += episode.run()
        scores.append(episode.score)

    for param in net.parameters():
        param.grad /= ct

    optimizer.step()

    net.save()
    print(f'epoch {epoch}: [{min(scores):5.2f}, {(sum(scores) / len(scores)):5.2f}, {max(scores):5.2f}], {time.time() - t}s')

