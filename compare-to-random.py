#!/usr/bin/env python3

DISPLAY = False
SCALE = 5

import random

import matplotlib.pyplot as plt

from agent import Agent
from game import Direction, Game

class RandomBot:
    def __init__(self): pass

    def get_move(self, game):
        move_mask = game.get_available_moves()
        available_moves = [k for k, v in enumerate(move_mask) if v]
        if available_moves == []:
            return None
        move_index = random.choice(available_moves)
        return Direction(move_index + 1)
        

def gameplay_hist(bot):
    scores = []
    for _ in range(2**SCALE):
        game = Game(exp=True)
        if DISPLAY: print(game)
        while (direction := bot.get_move(game)) is not None:
            game.move(direction)

            if DISPLAY:
                print(f'direction: {direction}')
                print()
                print(f'score: {game.score}')
                print(game)

        scores.append(game.score)
    return scores


if __name__ == '__main__':
    # set up game playing agents
    rando = RandomBot()
    learned = Agent.load()

    # play games
    rando_scores = gameplay_hist(rando)
    learned_scores = gameplay_hist(learned)

    plt.hist(rando_scores, alpha=0.5, label='random', bins=SCALE)
    plt.hist(learned_scores, alpha=0.5, label='learned policy', bins=SCALE)
    plt.legend()
    plt.title(f'Scores (out of {2**SCALE} games)')
    plt.show()


