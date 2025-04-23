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

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Génération des obstacles et de l'objectif au démarrage
def generate_fixed_world():
    # Position de départ fixe
    start_pos = np.array([0.0, 0.0], dtype=np.float32)
    
    # objectif générée une fois et de manière aléatoire
    while True:
        goal = np.random.uniform(-9, 9, 2).astype(np.float32)
        if np.linalg.norm(goal - start_pos) > 2:
            break

    # obstacles générés une fois et de manière aléatoire
    obstacles = []
    num_obstacles = np.random.randint(3, 6)
    for _ in range(num_obstacles):
        while True:
            obstacle = np.random.uniform(-9, 9, 2).astype(np.float32)
            if (np.linalg.norm(obstacle - start_pos) > 2 and
                np.linalg.norm(obstacle - goal) > 2):
                obstacles.append(obstacle)
                break
    
    return goal, obstacles

FIXED_GOAL, FIXED_OBSTACLES = generate_fixed_world()


# Environnement Ackermann avec visualisation optionnelle
class AckermannEnv(gym.Env):
    def __init__(self, render=False):
        super(AckermannEnv, self).__init__()
        # Espace d'observation : position (x, y) et orientation (theta)
        self.observation_space = spaces.Box(low=np.array([-10, -10, -np.pi], dtype=np.float32),
                                            high=np.array([10, 10, np.pi], dtype=np.float32),
                                            dtype=np.float32)
        # Espace d'action : vitesse (v) et angle de braquage (delta)
        self.action_space = spaces.Box(low=np.array([0, -1], dtype=np.float32),
                                       high=np.array([1, 1], dtype=np.float32),
                                       dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float32)  # [x, y, theta]
        self.agent_radius = 0.5  # Rayon de l'agent
        self.goal_radius = 0.3  # Rayon de l'objectif
        self.render_active = render  # Activation de la visualisation

        # Utilisation des valeurs fixes générées une seule fois
        self.goal = FIXED_GOAL
        self.obstacles = FIXED_OBSTACLES

        self.path = self.generate_path()  # Chemin planifié
        self.path_index = 0
        self.reached_points = set()
        self.previous_distance_to_goal = np.linalg.norm(self.state[:2] - self.goal)

        # Initialisation de la visualisation uniquement si render=True
        if self.render_active:
            self.fig, self.ax = plt.subplots()
            self.agent_plot, = self.ax.plot([], [], 'bo', ms=10, label='Agent')
            self.path_colors = ['purple'] * len(self.path)
            if self.path:
                self.path_plot = self.ax.scatter(*zip(*self.path), c=self.path_colors, s=20, label='Chemin')
            else:
                print("Avertissement : Aucun chemin trouvé vers l'objectif.")
                self.path_plot = self.ax.scatter([], [], c=[], s=20, label='Chemin')
            self.goal_plot = self.ax.scatter(*self.goal, c='green', s=100, label='Objectif')
            self.obstacle_plot = self.ax.scatter(*zip(*self.obstacles), c='red', s=100, label='Obstacles')
            self.trajectory = []
            self.trajectory_plot, = self.ax.plot([], [], 'b-', linewidth=1, label='Trajectoire')
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.legend()
            plt.title("Simulation de l'Agent Ackermann")

    # calcul de la distance euclidienne entre deux points
    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    # fonction astar qui calcul la trajectoire à suivre
    def astar(self, start, goal):
        open_set = [] # Liste contenant les noeuds que l'on va explorer
        heapq.heappush(open_set, (0, start)) # ajoute le premier noeuds avec 0 comme valeur    
        came_from = {} # dictionnaire stockant les noeuds précédents
        g_score = {start: 0} # Initialise et stock le coût réel entre le pt de départ et chaque noeuds 
        f_score = {start: self.heuristic(start, goal)} # calcul le coût estimé entre chaque noeuds et l'arrivée
        while open_set: # tant qu'il exite des noeuds 
            _, current = heapq.heappop(open_set) # cherche le nouds le plus proche
            if self.heuristic(current, goal) < self.goal_radius: # vérifie si on à atteint l'arrivée
                path = [] # liste entre le départ et le noeuds actuel 
                while current in came_from: # tant que on atteint pas le départ
                    path.append(current) # on ajout le noeud actuel
                    current = came_from[current] # mise a jour en prenant le noeud précédent
                path.reverse() # on inverse pour avoir la liste dans le bon ordre
                return path
            for neighbor in self.get_neighbors(current): # initialisation des noeuds suivants
                tentative_g_score = g_score[current] + 1 # calcul du coût
                # on va vérifier si le voisin n'as pas été atteint et si aucun autre chemin n'est plus court
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current # mise à jour des noeuds
                    g_score[neighbor] = tentative_g_score # mise à jour du score réel
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal) # mise à jour du score total
                    heapq.heappush(open_set, (f_score[neighbor], neighbor)) # ajout du voisin avec son score dans la liste
        return [] # retourne liste vide si aucun chemin trouvé

    # création des noeuds voisins valide pour notre chemin
    def get_neighbors(self, node):
        x, y = node
        step_size = 0.2 # pas de 0.2
        # créeations d'une liste coomportant les voisins possible dans les 8 directions autour du noeuds actuel
        neighbors = [(x + step_size, y), (x - step_size, y), (x, y + step_size), (x, y - step_size),
                     (x + step_size, y + step_size), (x - step_size, y - step_size),
                     (x - step_size, y + step_size), (x + step_size, y - step_size)]
        valid_neighbors = [] # on créer notre liste qui va stockés les voisins valides
        safe_distance_path = 1.8 # distance minimal avec les obstacle de 1.8
        for nx, ny in neighbors: # on vérifié si les voisins sont assez proches et si ils ne rentrent pas en colisions avec des obstacles
            if -10 <= nx <= 10 and -10 <= ny <= 10 and not self.is_collision(nx, ny, safe_distance_path):
                valid_neighbors.append((nx, ny)) # ajout si conditions vérifées
        return valid_neighbors

    # fonction si il y a des collisions avec un obstacle
    def is_collision(self, x, y, safe_distance):
        for ox, oy in self.obstacles: #coord de l'obstacle
            if math.sqrt((x - ox)**2 + (y - oy)**2) < safe_distance + self.agent_radius: # si distance entre les deux centre inférieur au rayon de l'obstacle
                return True # alors collision
        return False

    # lissage de la courbe pour le comportement ackermann de l'agent
    def smooth_path(self, path):
        if len(path) < 3: #chemin de plus de 3 pts
            return path
        x, y = zip(*path) # permet de décomposer les tuples en liste de x et de y
        t = range(len(path))
        # CubicSpline permet de lisser en créeant une courbe passant par les centre
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        t_new = np.linspace(0, len(path) - 1, len(path) * 2) # creation des nouveaux points a paritr du lissage des deux courbes précédentes
        return list(zip(cs_x(t_new), cs_y(t_new))) # interpolation des point et retourne le nouveau chemin avec 2 fois plus de points

    # génère la trajectoire finale
    def generate_path(self):
        start = (self.state[0], self.state[1]) # position de l'agent
        goal = (self.goal[0], self.goal[1]) # position de l'arrivé
        raw_path = self.astar(start, goal) # utilise l'algo astar
        return self.smooth_path(raw_path) # utilise la fonction permettant de lisser le chemin

    # fonction pour reset apres chaque tentative du drone
    def reset(self):
        self.state = np.zeros(3, dtype=np.float32)
        self.path_index = 0
        self.start_time = time.time()
        self.points_reached = 0
        self.reached_points.clear()
        self.trajectory = []
        self.previous_distance_to_goal = np.linalg.norm(self.state[:2] - self.goal)
        if self.render_active:
            self.path_colors = ['purple'] * len(self.path)
            self.path_plot.set_facecolors(self.path_colors)
        return self.state

    # trouve le point le plus proche
    def find_closest_point(self):
        if self.path_index < len(self.path):
            return self.path[self.path_index]
        return None

    # permet de definir le comportement et action du bateau pendant la simulation
    def step(self, action):
        v, delta = action
        v = max(min(v, 0.3), 0.1)  # Limiter la vitesse
        delta = max(min(delta, np.pi / 4), -np.pi / 4)  # Limiter l'angle
        previous_position = self.state[:2].copy()
        self.state[0] += v * np.cos(self.state[2])
        self.state[1] += v * np.sin(self.state[2])
        self.state[2] += v * np.tan(delta) / 1.0  # Longueur du bateau = 1.0
        reason = ""

        # Vérification des limites
        if not (-10 <= self.state[0] <= 10 and -10 <= self.state[1] <= 10):
            return self.state, -100, True, "Sortie des contours" # récompense de -100

        # Vérification du temps
        if time.time() - self.start_time > 5: # un essai pour 5 seconde max
            return self.state, -500, True, "Temps dépassé" # récompense de -500

        # Distance à l'objectif
        distance_to_goal = np.linalg.norm(self.state[:2] - self.goal)
        if distance_to_goal < self.agent_radius + self.goal_radius:
            return self.state, 500, True, "Objectif atteint !" # récompense de +500

        # Calcul de la récompense pour passer les points de la trajectoire
        closest_point = self.find_closest_point()
        if closest_point is None:
            reward = -distance_to_goal
            done = False
        else:
            distance_to_path = np.linalg.norm(self.state[:2] - closest_point)
            progression_reward = 5 * (self.previous_distance_to_goal - distance_to_goal)
            rotation_penalty = -0.5 if np.linalg.norm(self.state[:2] - previous_position) < 0.1 else 0
            reward = progression_reward + rotation_penalty - 2 * distance_to_path - 4 * distance_to_goal # pénalité si il tourne sur lui même
            done = False
            if distance_to_path < self.agent_radius + self.goal_radius:
                if self.path_index not in self.reached_points:
                    reward += 10  # +10 pour chaque points atteints
                    self.reached_points.add(self.path_index)
                    self.points_reached += 1
                    if self.render_active:
                        self.path_colors[self.path_index] = 'orange' # on change la couleur du point atteint en orange
                self.path_index += 1
                if self.path_index >= len(self.path):
                    reward += 500 # si dernier points atteint (arrivé) +500 
                    done = True
                    reason = "Objectif atteint !"

        self.previous_distance_to_goal = distance_to_goal

        # Vérification des collisions
        for obstacle in self.obstacles:
            if np.linalg.norm(self.state[:2] - obstacle) < self.agent_radius + self.goal_radius:
                return self.state, -200, True, "Collision avec obstacle"

        # Mise à jour de la visualisation
        if self.render_active:
            self.trajectory.append((self.state[0], self.state[1]))
            self.draw_agent()

        return self.state, reward, done, reason

    # on dessine l'agent tel un rond
    def draw_agent(self):
        self.agent_plot.set_data([self.state[0]], [self.state[1]])
        if len(self.trajectory) > 1:
            traj_x, traj_y = zip(*self.trajectory)
            self.trajectory_plot.set_data(traj_x, traj_y)
        self.path_plot.set_facecolors(self.path_colors)
        plt.pause(0.01)

    def close(self):
        if self.render_active:
            plt.close()


# on utilise la lybrairie PyTorch pour créer nos réseau de neuronnes

# 1er réseau de neuronne : l'Acteur permet de choisir les action a effectuer
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x): # Les états x est en entrées
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x # retourne les actions prédit par le réseau

# 2ème réseau de neuronne : le Critique permet d'évaluer et réduire la fonction Q prenant en compte les récompenses
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x, u): # les états x et actions u sont des entrées
        # on va calculer la fonction Q en fonction des entrées 
        x = torch.relu(self.fc1(torch.cat([x, u], 1)))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x # retourne la valeur Q

# Algo de DDPG
class DDPG:
    # Crétion de réseau copie pour l'acteur et le critique permettant une stabilité de l'apprentissage
    def __init__(self, state_dim, action_dim): 
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        # on va copier les valeurs des poids dans les resaux principaux
        self.actor_target.load_state_dict(self.actor.state_dict())  
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001) # définit le taux d'apprentissage du réseau 1
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001) # définit le taux d'apprentissage du réseau 2
        self.replay_buffer = deque(maxlen=10000) # on va stocker les expériences  (jusqu'a 10000 par défaut)
        self.gamma = 0.99 # hypperparametre pour l'apprentissage
        self.tau = 0.005

    def select_action(self, state, episode): # definit l'acteur comme le resau effectuant l'action
        # en convertit le tenseur PyTorch en tableau numpy pour l'exploitation
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        action[0] = max(min(action[0], 0.3), 0.1) # limite la vitesse entre 0.1 et 0.3
        action[1] = max(min(action[1], np.pi / 4), -np.pi / 4) # limite l'angle de braquage à +/- 45°
        noise_scale = max(0.05, 0.1 * (0.995 ** episode)) # ajout d'un bruit gaussien pour l'exploration 
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action + noise, [0, -1], [1, 1])
        return action # retourne l'action avec le bruit

    # fonction qui va entrainer (faire une boucle rétroactive entre les réseaux)
    def train(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size) # on prend un échantillon aléatoire
        state, action, reward, next_state, done = zip(*batch) # on décompose les tuples
        # on va convertir les données en tenseurs PyTorch
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(np.array(done, dtype=np.float32)).unsqueeze(1).to(device)

        # initialisation de la fonction maximisant  la meilleure action avec le critic
        target_q = self.critic_target(next_state, self.actor_target(next_state)) # l'acteur va prédire l'action et le critic va l'évaluer
        target_q = reward + (1 - done) * self.gamma * target_q # Equation de Bellman
        current_q = self.critic(state, action) # le critique principale estime l'action actuelle
        critic_loss = nn.MSELoss()(current_q, target_q.detach()) # on calcul la différence en la prédiction de l'action effectué et la prochaine (erreur quadratique moyenne)

        actor_loss = -self.critic(state, self.actor(state)).mean() # on calcul la perte en tant qu'opposé de la valeur Q

        # ici on optimise/met à jour le critic
        self.critic_optimizer.zero_grad() # réinitialisation des gradient
        critic_loss.backward() # on calcul les gradient de la perte
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) #norme du gradient à 1
        self.critic_optimizer.step() # on met à jour les poids du critic

        # ici on optimise/met à jour l'acteur
        self.actor_optimizer.zero_grad() # on reinitialise les gradients
        actor_loss.backward() # on les calcul
        self.actor_optimizer.step() #on met à jour les poids

        # on met à jour les poids des réseaux cible avec une combinaison linéaire
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # on sauvegarde les données dans un buffer pour ensuite venir le reprendre pendant l'entrainement
    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # sauvegarde les poids des réseaux de neuronnes dans des fichiers
    def save(self, filename):
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")

# Entraînement
env = AckermannEnv(render=False)
agent = DDPG(state_dim=3, action_dim=2) # 3 dimension pour l'état (x,y,z) et 2 pour les actions (vitesse et angle de braquage) 
num_episodes = 2000
batch_size = 20 # 20 essais différents pour l'entrainement (64 par défaut)
eval_interval = 1 # nombre de fois que l'on veut visualiser notre agent

# paramétrage pour chaque épisode
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    # on va effectuer les actions et stocker les valeurs dans un buffer et cela pendant 30 épisodes
    while not done and step_count < 30:
        action = agent.select_action(state, episode)
        next_state, reward, done, reason = env.step(action)
        agent.add_to_replay_buffer(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step_count += 1
        if len(agent.replay_buffer) > batch_size: # si on a effectuer plus de 30 épisodes, alors on entraine l'agent en piochant dans notre batch
            agent.train(batch_size)

    # affichage
    if step_count >= 30:
        print(f'Episode {episode}, Total Reward: {total_reward:.2f}, Points Reached: {env.points_reached}, Reason: Temps dépassé')
    else:
        print(f'Episode {episode}, Total Reward: {total_reward:.2f}, Points Reached: {env.points_reached}, Reason: {reason}')

    # Évaluation permettant une visualisation de l'agent lors d'un entrainement
    if (episode + 1) % eval_interval == 10: # interval d'une évaluation, ici 10
        agent.save(f"ddpg_model_episode_{episode+1}") # on sauvegarde les poids que lors d'évaluation
        eval_env = AckermannEnv(render=True)
        state = eval_env.reset()
        eval_reward = 0
        done = False
        step_count = 0
        while not done and step_count < 30: # tant qu"on à pas assez fais de simulation on entraine pas donc on évalue pas, on accumule les récompnes
            action = agent.select_action(state, episode)
            state, reward, done, reason = eval_env.step(action)
            eval_reward += reward
            step_count += 1
        if step_count >= 30: # on affiche le résultat d'une évaluation tous les 10 à partir de 30 essais
            print(f'Evaluation at Episode {episode+1}, Eval Reward: {eval_reward:.2f}, Reason: Temps dépassé')
        else:
            print(f'Evaluation at Episode {episode+1}, Eval Reward: {eval_reward:.2f}, Reason: {reason}')
        eval_env.close()

env.close()