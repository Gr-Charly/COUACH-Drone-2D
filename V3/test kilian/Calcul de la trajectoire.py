import numpy as np
import tkinter as tk
import math
import random
import heapq


class AckermannAgent:
    def __init__(self, x=100, y=100, theta=0, v=40, L=50):
        self.x = x  # Position initiale en x
        self.y = y  # Position initiale en y
        self.theta = theta  # Orientation initiale (angle)
        self.v = v  # Vitesse du véhicule
        self.L = L  # Longueur de l'axe du véhicule
        self.delta = 0  # Angle de braquage des roues avant (initialement 0)

    def get_position(self):
        """Retourne la position actuelle de l'agent."""
        return self.x, self.y, self.theta

    def distance_to_goal(self, goal_x, goal_y):
        """Retourne la distance entre l'agent et l'objectif."""
        return math.sqrt((self.x - goal_x) ** 2 + (self.y - goal_y) ** 2)


class Environment:
    def __init__(self, width=500, height=500, num_obstacles=5, start_margin=50, goal_margin=50):
        self.width = width
        self.height = height
        self.start_margin = start_margin  # Distance minimale entre le départ et un obstacle
        self.goal_margin = goal_margin  # Distance minimale entre l'arrivée et un obstacle

        # Position du point de départ et d'arrivée
        self.agent = AckermannAgent(x=width // 4, y=height // 4, theta=0, v=40, L=100)
        self.goal_x = width - 100
        self.goal_y = height - 100

        # Générer des obstacles sans interférer avec le départ et l'arrivée
        self.obstacles = self.generate_random_obstacles(num_obstacles)

        self.safe_distance = 30

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

    def generate_random_obstacles(self, num_obstacles):
        obstacles = []
        while len(obstacles) < num_obstacles:
            ox = random.randint(50, self.width - 50)
            oy = random.randint(50, self.height - 50)
            size = random.randint(10, 30)  # Taille des obstacles

            if self.is_valid_obstacle(ox, oy, size):
                obstacles.append((ox, oy, size))
        return obstacles

    def is_valid_obstacle(self, ox, oy, size):
        """Vérifie que l'obstacle ne se superpose pas à l'objectif ni au point de départ."""
        distance_to_start = math.sqrt((ox - self.agent.x) ** 2 + (oy - self.agent.y) ** 2)
        distance_to_goal = math.sqrt((ox - self.goal_x) ** 2 + (oy - self.goal_y) ** 2)

        # Vérifier que l'obstacle est à une distance suffisante du point de départ et de l'arrivée
        if distance_to_start > self.start_margin + size and distance_to_goal > self.goal_margin + size:
            return True
        return False

    def heuristic(self, a, b):
        """Distance Euclidienne entre deux points"""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def astar(self, start, goal):
        """Implémentation de l'algorithme A*"""
        open_set = []
        heapq.heappush(open_set, (0, start))  # Priorité = coût
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruire le chemin
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

        return []  # Pas de chemin trouvé

    def get_neighbors(self, node):
        """Retourne les voisins d'un nœud avec une résolution plus fine"""
        x, y = node
        neighbors = []
        step_size = 5  # Réduction de la taille d'un pas (plus petit que 10)

        # Essayer de se déplacer dans toutes les directions autour du point actuel
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue  # Ignorer la position actuelle

                nx, ny = x + dx, y + dy

                # Vérifier que le voisin est dans les limites de la fenêtre et qu'il n'y a pas d'obstacle
                if 0 <= nx < self.width and 0 <= ny < self.height and not self.is_collision(nx, ny):
                    neighbors.append((nx, ny))

        return neighbors

    def is_collision(self, x, y):
        """Vérifie s'il y a une collision avec un obstacle"""
        for ox, oy, size in self.obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < size + self.safe_distance:
                return True
        return False

    def get_euclidean_path(self, start, goal):
        """Retourne une liste de points formant une ligne droite entre le départ et l'arrivée."""
        path = []
        x1, y1 = start
        x2, y2 = goal

        # Calculer la pente de la ligne
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Calculer le nombre d'étapes pour couvrir la distance
        steps = int(distance / 10)  # On divise la distance par un facteur pour avoir des points d'étape

        for i in range(steps + 1):
            x = x1 + dx * (i / steps)
            y = y1 + dy * (i / steps)
            path.append((x, y))

        return path

    def draw_agent_and_goal(self):
        self.canvas.delete("all")
        x, y, theta = self.agent.get_position()
        self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="blue")
        self.canvas.create_oval(self.goal_x - 10, self.goal_y - 10, self.goal_x + 10, self.goal_y + 10, fill="green")

        for ox, oy, size in self.obstacles:
            self.canvas.create_oval(ox - size, oy - size, ox + size, oy + size, fill="red")

    def draw_path(self, path, color):
        """Dessine le chemin calculé"""
        for (x, y) in path:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color)

    def update(self):
        start = (self.agent.x, self.agent.y)
        goal = (self.goal_x, self.goal_y)

        # Trajectoire Euclidienne
        euclidean_path = self.get_euclidean_path(start, goal)

        # Trajectoire optimale (A*)
        optimal_path = self.astar(start, goal)

        if not optimal_path:
            print("Aucun chemin trouvé.")
        else:
            print(f"Chemin trouvé : {optimal_path}")

        self.draw_agent_and_goal()
        self.draw_path(euclidean_path, "blue")  # Dessiner la trajectoire Euclidienne
        self.draw_path(optimal_path, "purple")  # Dessiner la trajectoire optimale

        if self.agent.distance_to_goal(self.goal_x, self.goal_y) < 10:
            print("L'agent a atteint l'objectif !")
            return

        self.root.after(10, self.update)

    def start(self):
        self.update()
        self.root.mainloop()


if __name__ == "__main__":
    env = Environment(num_obstacles=5, start_margin=50, goal_margin=50)  # Modifie les marges de sécurité
    env.start()
