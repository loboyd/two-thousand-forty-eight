#!/usr/bin/env python3

import enum

class Direction(enum.Enum):
    RIGHT = 1
    DOWN = 2
    UP = 3
    LEFT = 4

class Game:

    EMPTY_BOARD = [[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    def __init__(self, score=0, board=EMPTY_BOARD): (self.score, self.board) = (score, board)

    def move(self, direction):
        if direction == Direction.RIGHT:
            for r in range(4):
                self.board[r] = reversed(Game.move_unit(reversed(self.board[r])))
        if direction == Direction.LEFT:
            for r in range(4):
                self.board[r] = Game.move_unit(self.board[r])
        if direction == Direction.UP:
            for c in range(4):
                unit = [self.board[0][c], self.board[1][c], self.board[2][c], self.board[3][c]]
                unit = Game.move_unit(unit)
                self.board[0][c] = unit[0]
                self.board[1][c] = unit[1]
                self.board[2][c] = unit[2]
                self.board[3][c] = unit[3]
        if direction == Direction.DOWN:
            for c in range(4):
                unit = [self.board[3][c], self.board[2][c], self.board[1][c], self.board[0][c]]
                unit = Game.move_unit(unit)
                self.board[0][c] = unit[3]
                self.board[1][c] = unit[2]
                self.board[2][c] = unit[1]
                self.board[3][c] = unit[0]

    @staticmethod
    def move_unit(unit):
        unit = Game.collapse(unit)
        for i in range(3):
            if unit[i] > 0 and unit[i] == unit[i+1]:
                unit[i] += 1
                unit[i+1] = 0
        unit = Game.collapse(unit)
        return unit

    @staticmethod
    def collapse(unit):
        unit = [x for x in unit if 0 < x]
        unit += [0] * (4 - len(unit)) # pad the 0s on the right
        return unit

    def __repr__(self):
        return "\n".join(["  ".join("{:2}".format(x) for x in row) for row in self.board])


game = Game()
print(game)
game.move(Direction.RIGHT)
print()
print(game)

