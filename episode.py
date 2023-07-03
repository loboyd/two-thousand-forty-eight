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

class Batch:
    def __init__(self, agent, batch_size=1, gamma=1):
        self.agent = agent
        self.batch_size = batch_size
        self.gamma = gamma
        self.episodes = [Episode(agent) for _ in range(batch_size)]

    def run(self):
        games = [Game() for _ in range(self.batch_size)]

        while z := [(ind, game) for ind, game in enumerate(games) if game.ongoing()]:
            indices, in_play = zip(*z) # unzip (i don't understand how this works, but it does)
            moves = self.agent.get_batch_moves(in_play)
            for ind, game, move in zip(indices, in_play, moves):
                self.episodes[ind].states.append(copy.deepcopy(game))
                self.episodes[ind].actions.append(move)
                prev_score = game.score
                game.move(move)
                self.episodes[ind].rewards.append(game.score - prev_score)
                self.episodes[ind].score = game.score

        for ep in self.episodes:
            ep._adjust_rewards()

