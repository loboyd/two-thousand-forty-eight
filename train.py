#!/usr/bin/env python3

LOAD = False
SAVE = False
BATCH_SIZE=2**8

import random
import time

import torch

from agent import Agent, set_seed
from episode import Batch, Episode
from game import Game

seed = 41
random.seed(seed)
set_seed(seed)

# set up agent and optimizer
agent = Agent.load() if LOAD else Agent()
optimizer = torch.optimizers.Adam(agent.parameters(), lr=0.002)

while True:
    # run a batch of episodes
    batch = Batch(agent, batch_size=BATCH_SIZE, gamma=0.96)
    batch.run()

    # do an update
    agent.update(optimizer, batch.episodes)
    if SAVE: agent.save()

