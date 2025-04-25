import pygame
import random
import math
from enum import Enum
from collections import namedtuple
import numpy as np
import os

pygame.init()
font_path = os.path.join(os.path.dirname(__file__), 'arial.ttf')
font = pygame.font.Font(font_path, 25)

class Direction(Enum):
    FORWARD = 1
    LEFT = 2
    RIGHT = 3

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class AckermannAgent:
    def __init__(self, x=100, y=100, theta=0, v=0, L=10):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v  # Vitesse initiale
        self.L = L
        self.delta = 0  # Angle de braquage initial
        self.max_speed = 40  # Vitesse maximale
        self.min_speed = 0  # Vitesse minimale (recul)
        self.acceleration = 2  # Accélération
        self.deceleration = 5  # Décélération

    def update_position(self, delta, dt=0.1):
        self.delta = delta

        # Mettre à jour la vitesse en fonction de l'accélération ou de la décélération
        if self.v < self.max_speed:
            self.v += self.acceleration * dt
        if self.v > self.max_speed:
            self.v = self.max_speed
        if self.v < self.min_speed:
            self.v = self.min_speed

        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += (self.v / self.L) * math.tan(self.delta) * dt

    def get_position(self):
        return self.x, self.y, self.theta

    def distance_to_goal(self, goal_x, goal_y):
        return math.sqrt((self.x - goal_x) ** 2 + (self.y - goal_y) ** 2)

class BoatGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Boat Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, regenerate=True):
        self.boat = AckermannAgent(x=self.w / 2, y=self.h / 2)
        self.score = 0
        if regenerate:
            self.initial_path = self._generate_path()
            self.path = self.initial_path.copy()
            self.obstacles = self._generate_obstacles()
        else:
            self.path = self.initial_path.copy()
        self.frame_iteration = 0
        self.current_goal = self._find_closest_goal()
        self.previous_distance = self.boat.distance_to_goal(self.current_goal.x, self.current_goal.y)

    def _generate_path(self):
        path = []
        for _ in range(5):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            path.append(Point(x, y))
        return path

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(0):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            obstacles.append(Point(x, y))
        return obstacles

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.boat.update_position(self.boat.delta)

        reward = 0
        game_over = False

        # Vérifier les collisions
        if self.is_collision() or self.frame_iteration > 100 * len(self.path):
            game_over = True
            reward = -100  # Pénalité importante pour les collisions
            return reward, game_over, self.score

        # Calculer la récompense de proximité pour le point cible le plus proche
        current_distance = self.boat.distance_to_goal(self.current_goal.x, self.current_goal.y)
        reward = (self.previous_distance - current_distance) * 10  # Récompense continue basée sur la diminution de la distance

        # Pénalité pour la proximité des obstacles
        for obs in self.obstacles:
            if self.boat.distance_to_goal(obs.x, obs.y) < BLOCK_SIZE * 2:
                reward -= 50  # Pénalité pour être trop proche d'un obstacle

        # Mettre à jour le score et l'objectif si le point est atteint
        if current_distance < BLOCK_SIZE:
            self.score += 1
            reward += 200 +  (100 / (current_distance + 0.1))# Récompense supplémentaire pour atteindre un point
            self.path.remove(self.current_goal)
            if self.path:
                self.current_goal = self._find_closest_goal()
                self.previous_distance = self.boat.distance_to_goal(self.current_goal.x, self.current_goal.y)

        self.previous_distance = current_distance

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _find_closest_goal(self):
        boat_x, boat_y, _ = self.boat.get_position()
        closest_goal = min(self.path, key=lambda goal: self.boat.distance_to_goal(goal.x, goal.y))
        return closest_goal

    def _update_ui(self):
        self.display.fill(BLACK)
        x, y, _ = self.boat.get_position()
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE2, pygame.Rect(x + 4, y + 4, 12, 12))

        for pt in self.path:
            if pt == self.current_goal:
                # Dessiner le point le plus proche en vert
                pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            else:
                # Dessiner les autres points en rouge
                pygame.draw.rect(self.display, RED, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.obstacles:
            pygame.draw.rect(self.display, (100, 100, 100), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        max_delta = 0.5  # Valeur maximale pour l'angle de braquage
        delta_increment = 0.1  # Incrément pour ajuster l'angle de braquage

        if np.array_equal(action, [1, 0, 0]):
            # Accélérer tout droit
            self.boat.delta = 0
        elif np.array_equal(action, [0, 1, 0]):
            # Tourner à gauche
            self.boat.delta = max(self.boat.delta - delta_increment, -max_delta)
        elif np.array_equal(action, [0, 0, 1]):
            # Tourner à droite
            self.boat.delta = min(self.boat.delta + delta_increment, max_delta)

        self.boat.update_position(self.boat.delta)

    def is_collision(self, pt=None):
        if pt is None:
            boat_x, boat_y, _ = self.boat.get_position()
            pt = Point(boat_x, boat_y)

        if pt.x > self.w or pt.x < 0 or pt.y > self.h or pt.y < 0:
            return True
        if any(self.boat.distance_to_goal(obs.x, obs.y) < BLOCK_SIZE for obs in self.obstacles):
            return True
        return False
