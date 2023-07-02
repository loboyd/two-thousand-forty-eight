import copy

from game import Game

class Episode:
    def __init__(self, agent, gamma=1):
        self.agent = agent
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []
        self.score = 0

    def run(self):
        game = Game(exp=True)
        prev_score = game.score
        while move := self.agent.get_move(game):
            self.states.append(copy.deepcopy(game))
            self.actions.append(move)
            game.move(move)
            self.rewards.append(game.score - prev_score)

        self._adjust_rewards()
        self.score = game.score

    def _adjust_rewards(self):
        R = 0
        for i in range(len(self.rewards)-1, -1, -1):
            R = self.rewards[i] + self.gamma * R
            self.rewards[i] = R

