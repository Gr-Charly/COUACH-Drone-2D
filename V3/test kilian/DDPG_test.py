import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import heapq
from scipy.interpolate import CubicSpline

# Définition de l'environnement Ackermann
class AckermannEnv(gym.Env):
    def __init__(self):
        super(AckermannEnv, self).__init__()
        # Espace d'observation : [x, y, theta]
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -np.pi], dtype=np.float32),
            high=np.array([10, 10, np.pi], dtype=np.float32),
            dtype=np.float32
        )
        # Espace d'action : [vitesse, angle de direction]
        self.action_space = spaces.Box(
            low=np.array([0, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        # Initialisation de l'état : [x, y, theta]
        self.state = np.zeros(3, dtype=np.float32)
        self.goal = np.array([5, 5], dtype=np.float32)  # Objectif
        self.obstacles = [np.array([2, 2], dtype=np.float32), np.array([3, 3], dtype=np.float32)]  # Obstacles
        self.agent_radius = 0.5  # Rayon de l'agent
        self.goal_radius = 0.3   # Rayon de l'objectif

        # Générer la trajectoire une seule fois
        self.path = self.generate_path()
        self.path_index = 0

        # Initialisation de la visualisation avec Matplotlib
        self.fig, self.ax = plt.subplots()
        self.agent_plot, = self.ax.plot([], [], 'bo', ms=10)  # Agent en bleu
        self.path_plot = self.ax.scatter(*zip(*self.path), c='purple')  # Trajectoire en violet
        self.goal_plot = self.ax.scatter(*self.goal, c='green', s=100)  # Objectif en vert
        self.obstacle_plot = self.ax.scatter(*zip(*self.obstacles), c='red', s=100)  # Obstacles en rouge
        self.trajectory = []  # Liste pour stocker la trajectoire suivie
        self.trajectory_plot, = self.ax.plot([], [], 'b-', linewidth=1)  # Trace de la trajectoire
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

    def generate_path(self):
        """Génère une trajectoire optimisée de départ à l'objectif."""
        start = (self.state[0], self.state[1])
        goal = (self.goal[0], self.goal[1])
        raw_path = self.astar(start, goal)
        return self.smooth_path(raw_path)

    def heuristic(self, a, b):
        """Heuristique pour A* : distance euclidienne."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def astar(self, start, goal):
        """Algorithme A* pour trouver un chemin initial."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def get_neighbors(self, node):
        """Récupère les voisins valides pour A*."""
        x, y = node
        step_size = 0.2
        neighbors = [
            (x + step_size, y), (x - step_size, y),
            (x, y + step_size), (x, y - step_size),
            (x + step_size, y + step_size), (x - step_size, y - step_size),
            (x - step_size, y + step_size), (x + step_size, y - step_size)
        ]
        valid_neighbors = []
        safe_distance_path = 1.8
        for nx, ny in neighbors:
            if -10 <= nx <= 10 and -10 <= ny <= 10 and not self.is_collision(nx, ny, safe_distance_path):
                valid_neighbors.append((nx, ny))
        return valid_neighbors

    def is_collision(self, x, y, safe_distance):
        """Vérifie les collisions avec les obstacles."""
        for ox, oy in self.obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < safe_distance + self.agent_radius:
                return True
        return False

    def smooth_path(self, path):
        """Lisse la trajectoire avec des splines cubiques."""
        if len(path) < 3:
            return path
        x, y = zip(*path)
        t = range(len(path))
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        t_new = np.linspace(0, len(path) - 1, len(path) * 10)  # Plus de points pour une trajectoire fluide
        return list(zip(cs_x(t_new), cs_y(t_new)))

    def reset(self):
        """Réinitialise l'environnement."""
        self.state = np.zeros(3, dtype=np.float32)
        self.path_index = 0
        self.start_time = time.time()
        self.points_reached = 0
        self.trajectory = []  # Réinitialiser la trajectoire suivie
        return self.state

    def find_closest_point(self):
        """Trouve le point le plus proche sur la trajectoire."""
        if self.path_index < len(self.path):
            return self.path[self.path_index]
        return None

    def step(self, action):
        """Effectue une étape dans l'environnement."""
        v, delta = action
        v = max(min(v, 0.3), 0.1)  # Limite la vitesse
        delta = max(min(delta, np.pi / 4), -np.pi / 4)  # Limite l'angle
        self.state[0] += v * np.cos(self.state[2])
        self.state[1] += v * np.sin(self.state[2])
        self.state[2] += v * np.tan(delta) / 1.0  # Dynamique Ackermann

        reason = ""

        # Vérification des limites
        if not (-10 <= self.state[0] <= 10 and -10 <= self.state[1] <= 10):
            reward = -1
            done = True
            reason = "L'agent est sorti des contours."
            return self.state, reward, done, reason

        # Vérification du temps
        if time.time() - self.start_time > 5:
            closest_point = self.find_closest_point()
            if closest_point is not None:
                distance_to_goal = np.linalg.norm(self.state[:2] - np.array(closest_point))
                reward = -distance_to_goal
            else:
                reward = 0
            done = True
            reason = "Temps de simulation dépassé."
            return self.state, reward, done, reason

        # Calcul de la distance à l'objectif
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal)

        # Vérifier si l'agent a atteint l'objectif
        if distance_to_goal < self.agent_radius + self.goal_radius:
            reward = 50  # Récompense pour atteindre l'objectif
            done = True
            reason = "L'agent a atteint l'objectif !"
            self.path_index = len(self.path)  # Marquer la trajectoire comme terminée
        else:
            # Calcul de la récompense pour le suivi de la trajectoire
            closest_point = self.find_closest_point()
            distance_to_path = np.linalg.norm(self.state[:2] - closest_point)
            reward = -0.2 * distance_to_path - 0.1 * distance_to_goal + 1 / (distance_to_path + 0.1)

            if distance_to_path < self.agent_radius + self.goal_radius:
                reward += 10
                self.path_index += 1
                self.points_reached += 1
                if self.path_index >= len(self.path):
                    reward += 50
                    done = True
                    reason = "L'agent a atteint l'objectif !"
                    v = 0
                else:
                    done = False
            else:
                done = False

        # Vérification des collisions
        for obstacle in self.obstacles:
            if np.linalg.norm(self.state[:2] - obstacle) < self.agent_radius + self.goal_radius:
                reward -= 50
                done = True
                reason = "L'agent a touché un obstacle."
                break

        self.trajectory.append((self.state[0], self.state[1]))  # Ajouter la position actuelle
        self.draw_agent()
        return self.state, reward, done, reason

    def draw_agent(self):
        """Met à jour la position de l'agent dans la visualisation."""
        self.agent_plot.set_data([self.state[0]], [self.state[1]])
        if len(self.trajectory) > 1:
            traj_x, traj_y = zip(*self.trajectory)
            self.trajectory_plot.set_data(traj_x, traj_y)
        plt.pause(0.01)

    def render(self, mode='human'):
        pass  # La visualisation est gérée dans step

    def close(self):
        """Ferme la fenêtre de visualisation."""
        plt.close()

# Définition de l'acteur pour DDPG
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

# Définition du critique pour DDPG
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x, u):
        x = torch.relu(self.fc1(torch.cat([x, u], 1)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Implémentation de DDPG
class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.95  # Facteur d'escompte
        self.tau = 0.005   # Mise à jour douce des cibles

    def select_action(self, state, episode):
        """Sélectionne une action avec bruit d'exploration."""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        action = action.detach().numpy()[0]
        action[0] = max(min(action[0], 0.3), 0.1)  # Limite la vitesse
        action[1] = max(min(action[1], np.pi / 4), -np.pi / 4)  # Limite l'angle

        noise_scale = 0.1 * (0.995 ** episode)  # Décroissance exponentielle du bruit
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action += noise
        action = np.clip(action, [0, -1], [1, 1])
        return action

    def train(self, batch_size):
        """Entraîne les réseaux Actor et Critic."""
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done, dtype=np.float32)).unsqueeze(1)

        # Mise à jour du Critic
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Mise à jour de l'Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Mise à jour douce des réseaux cibles
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

# Entraînement de l'agent
env = AckermannEnv()
agent = DDPG(state_dim=3, action_dim=2)

num_episodes = 1000
batch_size = 64

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, episode)
        next_state, reward, done, reason = env.step(action)
        agent.add_to_replay_buffer(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.train(batch_size)

    print(f'Episode {episode}, Total Reward: {total_reward}, Points Reached: {env.points_reached}, Reason: {reason}')

env.close()