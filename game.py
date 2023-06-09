import copy
import enum
import random
import sys
import termios
import tty

class Direction(enum.Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class Game:

    EMPTY_BOARD = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    def __init__(self, score=0, board=EMPTY_BOARD, exp=False):
        self.score = score
        self.board = copy.deepcopy(board)
        self.exp = exp
        if self.board == Game.EMPTY_BOARD:
            self._place_random()
            self._place_random()

    # todo: make this obey `self.exp`
    def get_max_tile(self): return 2**max([max(row) for row in self.board])

    def move(self, direction):
        snapshot = copy.deepcopy(self.board)
        if direction == Direction.RIGHT:
            for r in range(4):
                new, score_change = Game._move_unit(reversed(self.board[r]), self.exp)
                self.board[r] = list(reversed(new))
                self.score += score_change
        elif direction == Direction.LEFT:
            for r in range(4):
                new, score_change = Game._move_unit(self.board[r], self.exp)
                self.board[r] = new
                self.score += score_change
        elif direction == Direction.UP:
            for c in range(4):
                unit = [self.board[0][c], self.board[1][c], self.board[2][c], self.board[3][c]]
                new, score_change = Game._move_unit(unit, self.exp)
                self.board[0][c] = new[0]
                self.board[1][c] = new[1]
                self.board[2][c] = new[2]
                self.board[3][c] = new[3]
                self.score += score_change
        elif direction == Direction.DOWN:
            for c in range(4):
                unit = [self.board[3][c], self.board[2][c], self.board[1][c], self.board[0][c]]
                new, score_change = Game._move_unit(unit, self.exp)
                self.board[0][c] = new[3]
                self.board[1][c] = new[2]
                self.board[2][c] = new[1]
                self.board[3][c] = new[0]
                self.score += score_change

        if self.board != snapshot:
            self._place_random()

    def ongoing(self):
        return self.get_move_mask() != [False] * 4

    def get_move_mask(self):
        moves = set()
        for r, c in [(x, y) for x in range(4) for y in range(3)]:
            if self.board[r][c] == self.board[r][c+1] != 0: moves.update([Direction.LEFT, Direction.RIGHT])
            if self.board[r][c] == 0 and self.board[r][c+1] > 0: moves.add(Direction.LEFT)
            if self.board[r][c] > 0 and self.board[r][c+1] == 0: moves.add(Direction.RIGHT)
        for r, c in [(x, y) for x in range(3) for y in range(4)]:
            if self.board[r][c] == self.board[r+1][c] != 0: moves.update([Direction.UP, Direction.DOWN])
            if self.board[r][c] == 0 and self.board[r+1][c] > 0: moves.add(Direction.UP)
            if self.board[r][c] > 0 and self.board[r+1][c] == 0: moves.add(Direction.DOWN)

        return [
            Direction.UP in moves,
            Direction.RIGHT in moves,
            Direction.DOWN in moves,
            Direction.LEFT in moves
        ]

    def _find_empties(self):
        """Return the index-pairs of all empty board positions"""
        return [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]

    def _place_random(self):
        empties = self._find_empties()
        if empties != []:
            r, c = random.choice(empties)
            self.board[r][c] = 1 if random.random() < 0.9 else 2

    @staticmethod
    def _move_unit(unit, exp=False):
        score_change = 0
        unit = Game._collapse(unit)
        for i in range(3):
            if unit[i] > 0 and unit[i] == unit[i+1]:
                unit[i] += 1
                score_change += 2**unit[i] # todo does this need to be `2**(unit[i]-1)`?
                unit[i+1] = 0
        unit = Game._collapse(unit)
        return unit, score_change

    @staticmethod
    def _collapse(unit):
        unit = [x for x in unit if 0 < x]
        unit += [0] * (4 - len(unit)) # pad the 0s on the right
        return unit

    def _tile_repr(self, tile): return f'{(2**tile if tile > 0 and self.exp else tile):>4}'

    def __repr__(self):
        return "\n".join(["  ".join(self._tile_repr(x) for x in row) for row in self.board])


def play_game():
    def _get_input_char():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        input_char = sys.stdin.read(1)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return input_char

    game = Game(exp=True)
    for _ in range(100): print()
    print(game)
    while True:
        move = _get_input_char()
        if move == 'j':
            game.move(Direction.LEFT)
        if move == 'k':
            game.move(Direction.DOWN)
        if move == 'l':
            game.move(Direction.RIGHT)
        if move == 'i':
            game.move(Direction.UP)
        if move == 'q':
            exit()
        for _ in range(100): print()
        print(game)
        print(game.get_move_mask())

