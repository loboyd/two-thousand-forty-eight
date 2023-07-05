#!/usr/bin/env python3

DISPLAY = False
SCALE = 8

import random
import time

import matplotlib.pyplot as plt

from agent import Agent
from episode import Batch
from game import Direction

class RandomBot:
    def __init__(self): pass

    def get_batch_moves(self, games, train=False):
        moves = []
        for game in games:
            move_mask = game.get_move_mask()
            available_moves = [k for k, v in enumerate(move_mask) if v]
            if available_moves == []:
                return None
            move_index = random.choice(available_moves)
            moves.append(Direction(move_index + 1))
        return moves
        

def gameplay_hist(bot):
    batch = Batch(bot, batch_size=2**SCALE)
    batch.run(train=False)
    return [ep.score for ep in batch.episodes]


if __name__ == '__main__':
    # set up game playing agents
    rando = RandomBot()
    learned = Agent.load()

    # play games
    t = time.time()
    rando_scores = gameplay_hist(rando)
    print(f'played RandomBot games in {time.time() - t} seconds')

    t = time.time()
    learned_scores = gameplay_hist(learned)
    print(f'played Agent games in {time.time() - t} seconds')

    plt.hist(rando_scores, alpha=0.5, label='random', bins=SCALE, density=True)
    plt.hist(learned_scores, alpha=0.5, label='learned policy', bins=SCALE, density=True)
    plt.legend()
    plt.title(f'Scores (out of {2**SCALE} games)')
    plt.show()


