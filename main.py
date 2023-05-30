#!/usr/bin/env python3

import copy
import enum
import random
import sys
import termios
import tty

class Direction(enum.Enum):
    RIGHT = 1
    DOWN = 2
    UP = 3
    LEFT = 4

class State(enum.Enum):
    ONGOING = 1
    OVER = 2

class Game:

    EMPTY_BOARD = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    def __init__(self, score=0, board=EMPTY_BOARD, state=State.ONGOING, exp=False):
        self.score = score
        self.board = board
        self.state = state
        self.exp = exp
        self._place_random()
        self._place_random()

    def move(self, direction):
        snapshot = copy.deepcopy(self.board)
        if direction == Direction.RIGHT:
            for r in range(4):
                self.board[r] = list(reversed(Game._move_unit(reversed(self.board[r]))))
        if direction == Direction.LEFT:
            for r in range(4):
                self.board[r] = Game._move_unit(self.board[r])
        if direction == Direction.UP:
            for c in range(4):
                unit = [self.board[0][c], self.board[1][c], self.board[2][c], self.board[3][c]]
                unit = Game._move_unit(unit)
                self.board[0][c] = unit[0]
                self.board[1][c] = unit[1]
                self.board[2][c] = unit[2]
                self.board[3][c] = unit[3]
        if direction == Direction.DOWN:
            for c in range(4):
                unit = [self.board[3][c], self.board[2][c], self.board[1][c], self.board[0][c]]
                unit = Game._move_unit(unit)
                self.board[0][c] = unit[3]
                self.board[1][c] = unit[2]
                self.board[2][c] = unit[1]
                self.board[3][c] = unit[0]
        if self.board != snapshot:
            self._place_random()

    def _place_random(self):
        empties = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if empties == []:
            self.state = State.OVER
            return
        r, c = random.choice(empties)
        self.board[r][c] = 1 if random.random() < 0.9 else 2

    @staticmethod
    def _move_unit(unit):
        unit = Game._collapse(unit)
        for i in range(3):
            if unit[i] > 0 and unit[i] == unit[i+1]:
                unit[i] += 1
                unit[i+1] = 0
        unit = Game._collapse(unit)
        return unit

    @staticmethod
    def _collapse(unit):
        unit = [x for x in unit if 0 < x]
        unit += [0] * (4 - len(unit)) # pad the 0s on the right
        return unit

    def _tile_repr(self, tile): return f'{(2**tile if tile > 0 and self.exp else tile):>4}'

    def __repr__(self):
        return "\n".join(["  ".join(self._tile_repr(x) for x in row) for row in self.board])


def get_input_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    input_char = sys.stdin.read(1)
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return input_char

game = Game()
print(game)
game.move(Direction.RIGHT)
print()
print(game)

