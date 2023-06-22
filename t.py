#!/usr/bin/env python3

FANOUT = 2**4

import copy

import torch

from agent import Agent
from game import Direction, Game

net = Agent.load()

# some mid-game state
#game = Game(exp=True)
#game.board = [
#    [2, 2, 2, 2],
#    [0, 3, 3, 3],
#    [0, 0, 4, 4],
#    [1, 0, 0, 5],
#]
#game.score = 272

def search(game):
    start_score = game.score
    score_gains = [0, 0, 0, 0]

    # play some games
    for (move_index, allowed) in enumerate(game.get_move_mask()):
        if allowed:
            move = Direction(move_index + 1)
            for _ in range(FANOUT):
                # copy the board for a trial
                trial = copy.deepcopy(game)

                # make the initial move
                trial.move(move)

                # play the rest of the game
                while net.play_move(trial): pass

                # accumulate score gain
                #score_gains[move_index] += trial.score - start_score
                score_gains[move_index] += trial.score

    score_gains = [x / FANOUT for x in score_gains]

    #print(score_gains)

    highest_expected_score = max(score_gains)
    print(highest_expected_score)

    best_move_index = score_gains.index(highest_expected_score)
    return Direction(best_move_index + 1)

game = Game(exp=True)
print(game)
while game.get_move_mask() != [0, 0, 0, 0]:
    best_move = search(game)
    game.move(best_move)
    print()
    print(game.score)
    print(game)

