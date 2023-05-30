#!/usr/bin/env python3

import enum

class Game:

    EMPTY_BOARD = [[16, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    class Action(enum.Enum):
        RIGHT = 1
        DOWN = 2
        UP = 3
        LEFT = 4

    def __init__(self, score=0, board=EMPTY_BOARD): (self.score, self.board) = (score, board)

    @staticmethod
    def move_unit(unit):
        for i in range(3):
            if unit[i] > 0 and unit[i] == unit[i+1]:
                unit[i] += 1
                unit[i+1] = 0
        unit = [x for x in unit if 0 < x]
        unit += [0] * (4 - len(unit)) # pad the 0s on the right

    def __repr__(self):
        return "\n".join(["  ".join("{:2}".format(x) for x in row) for row in self.board])


game = Game()

