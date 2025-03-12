import torch
import random
import numpy as np
import math
from collections import deque
from game import BoatGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

from collections import namedtuple

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

Point = namedtuple('Point', 'x, y')

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.9  # Commencer avec une valeur plus élevée pour plus d'exploration
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(12, 256, 3)  # Ajuster la taille d'entrée
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        boat = game.boat
        boat_x, boat_y, boat_theta = boat.get_position()

        # Calculer les points autour du bateau pour vérifier les collisions potentielles
        point_l = Point(boat_x - BLOCK_SIZE * math.cos(boat_theta), boat_y - BLOCK_SIZE * math.sin(boat_theta))
        point_r = Point(boat_x + BLOCK_SIZE * math.cos(boat_theta), boat_y + BLOCK_SIZE * math.sin(boat_theta))
        point_f = Point(boat_x + BLOCK_SIZE * math.cos(boat_theta - math.pi / 2), boat_y + BLOCK_SIZE * math.sin(boat_theta - math.pi / 2))
        point_b = Point(boat_x - BLOCK_SIZE * math.cos(boat_theta - math.pi / 2), boat_y - BLOCK_SIZE * math.sin(boat_theta - math.pi / 2))

        # Vérifier les collisions potentielles
        danger_straight = game.is_collision(point_f)
        danger_right = game.is_collision(point_r)
        danger_left = game.is_collision(point_l)

        # Obtenir la direction actuelle
        dir_l = boat.delta < 0
        dir_r = boat.delta > 0
        dir_f = boat.delta == 0

        # Obtenir la position du prochain objectif
        if game.path:
            goal_x, goal_y = game.current_goal
            goal_left = goal_x < boat_x
            goal_right = goal_x > boat_x
            goal_up = goal_y < boat_y
            goal_down = goal_y > boat_y
        else:
            goal_left = goal_right = goal_up = goal_down = False

        # Ajouter la distance relative à l'objectif
        if game.path:
            distance_to_goal = boat.distance_to_goal(goal_x, goal_y)
        else:
            distance_to_goal = 0

        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_f,
            goal_left,
            goal_right,
            goal_up,
            goal_down,
            len(game.path) > 0,  # Vérifie s'il reste des points dans le chemin
            distance_to_goal  # Distance relative à l'objectif
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = BoatGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Réinitialiser le jeu avec les mêmes obstacles et objectifs initiaux
            game.reset(regenerate=False)
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
