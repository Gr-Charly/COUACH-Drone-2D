import numpy as np
import tkinter as tk
import math
import random
import heapq

class AckermannAgent:
    def __init__(self, x=100, y=100, theta=0, v=40, L=100): # Valeur par default
        self.x = x  # Position initiale en x
        self.y = y  # Position initiale en y
        self.theta = theta  # Orientation initiale (angle)
        self.v = v  # Vitesse du véhicule
        self.L = L  # Longueur de l'axe du véhicule
        self.delta = 0  # Angle de braquage des roues avant (initialement 0)

    def update_position(self, delta, dt=0.1):
        """
        Mise à jour de la position de l'agent en utilisant le modèle d'Ackermann.
        """
        self.delta = delta
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += (self.v / self.L) * math.tan(self.delta) * dt

    def get_position(self):
        return self.x, self.y, self.theta

    def distance_to_goal(self, goal_x, goal_y):
        return math.sqrt((self.x - goal_x) ** 2 + (self.y - goal_y) ** 2)

class Environment:
    def __init__(self, width=500, height=500, num_obstacles=5, grid_resolution=10):
        self.width = width
        self.height = height
        self.agent = AckermannAgent(x=width // 4, y=height // 4, theta=0, v=5, L=5) # Valeur choisi
        self.goal_x = width - 100
        self.goal_y = height - 100
        self.obstacles = self.generate_random_obstacles(num_obstacles)
        self.grid_resolution = grid_resolution  # Résolution de la grille pour A*
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        # Labels pour afficher les paramètres
        self.x_label = tk.Label(self.root, text=f"x: {self.agent.x}")
        self.x_label.pack()
        self.y_label = tk.Label(self.root, text=f"y: {self.agent.y}")
        self.y_label.pack()
        self.theta_label = tk.Label(self.root, text=f"theta: {self.agent.theta}")
        self.theta_label.pack()
        self.v_label = tk.Label(self.root, text=f"v: {self.agent.v}")
        self.v_label.pack()
        self.L_label = tk.Label(self.root, text=f"L: {self.agent.L}")
        self.L_label.pack()
        self.delta_label = tk.Label(self.root, text=f"delta: {self.agent.delta}")
        self.delta_label.pack()

        self.path_found = False  # Variable pour contrôler l'affichage du message "Chemin trouvé."
        self.path = []  # Liste pour stocker le chemin trouvé
        self.path_index = 0  # Indice du point actuel sur le chemin

    def generate_random_obstacles(self, num_obstacles):
        obstacles = [(294, 435, 13), (219, 274, 18), (234, 223, 16)]
        return obstacles

    def draw_agent_and_goal(self):
        self.canvas.delete("all")
        x, y, theta = self.agent.get_position()
        self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="blue")
        self.canvas.create_line(x, y, x + 20 * math.cos(theta), y + 20 * math.sin(theta), arrow=tk.LAST)
        self.canvas.create_oval(self.goal_x - 10, self.goal_y - 10, self.goal_x + 10, self.goal_y + 10, fill="green")
        for ox, oy, size in self.obstacles:
            self.canvas.create_oval(ox - size, oy - size, ox + size, oy + size, fill="red")

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def astar(self, start, goal):
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
        x, y = node
        step_size = self.grid_resolution  # Résolution fine du pas de recherche
        neighbors = [
            (x + step_size, y), (x - step_size, y), 
            (x, y + step_size), (x, y - step_size),
            (x + step_size, y + step_size), (x - step_size, y - step_size),
            (x - step_size, y + step_size), (x + step_size, y - step_size)
        ]
        valid_neighbors = []
        safe_distance_path = 30 # 15 minimum
        for nx, ny in neighbors:
            if 0 <= nx < self.width and 0 <= ny < self.height and not self.is_collision(nx, ny, safe_distance_path):
                valid_neighbors.append((nx, ny))
        return valid_neighbors

    def is_collision(self, x, y, safe_distance):
        for ox, oy, size in self.obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < size + safe_distance:
                return True
        return False

    def update(self):
        start = (self.agent.x, self.agent.y)
        goal = (self.goal_x, self.goal_y)

        # Vérification si l'agent sort de la fenêtre
        if not (0 <= self.agent.x <= self.width and 0 <= self.agent.y <= self.height):
            print("L'agent a quitté la fenêtre.")
            return

        # Vérification si l'agent touche un obstacle
        safe_distance_agent = 10 # Ne pas modifier
        if self.is_collision(self.agent.x, self.agent.y, safe_distance_agent):
            print("L'agent a touché un obstacle.")
            return

        # Si le chemin n'est pas trouvé, on le génère une fois
        if not self.path:
            self.path = self.astar(start, goal)

            if not self.path:
                print("Aucun chemin trouvé.")
            elif not self.path_found:
                print(f"Chemin trouvé.")
                self.path_found = True  # Marquer que le chemin a été trouvé

                # Garder un point sur deux du chemin
                self.path = self.path[::10]  

                # Définir l'angle initial vers le premier point
                first_point = self.path[0]
                dx = first_point[0] - self.agent.x
                dy = first_point[1] - self.agent.y
                self.agent.theta = math.atan2(dy, dx)  # Orientation vers le premier point
                print(f"Angle initial ajusté à {math.degrees(self.agent.theta):.2f}°")

        if self.path:
            if (self.goal_x, self.goal_y) not in self.path:
                self.path.append((self.goal_x, self.goal_y))  # Ajouter le point de l'objectif

            # Obtenir le point suivant dans le chemin
            current_point = self.path[self.path_index]

            # Calculer l'angle vers le prochain point
            dx = current_point[0] - self.agent.x
            dy = current_point[1] - self.agent.y
            target_angle = math.atan2(dy, dx)

            # Calculer l'angle de braquage nécessaire pour tourner vers l'objectif
            delta = target_angle - self.agent.theta
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi

            # Mettre à jour la position de l'agent
            self.agent.update_position(delta)

            # Mettre à jour les labels pour afficher les paramètres
            self.x_label.config(text=f"x: {self.agent.x:.2f}")
            self.y_label.config(text=f"y: {self.agent.y:.2f}")
            self.theta_label.config(text=f"theta: {math.degrees(self.agent.theta):.2f}°")
            self.v_label.config(text=f"v: {self.agent.v}")
            self.L_label.config(text=f"L: {self.agent.L}")
            self.delta_label.config(text=f"delta: {math.degrees(self.agent.delta):.2f}°")

            # Vérifier si l'agent a atteint le point actuel du chemin
            if self.agent.distance_to_goal(current_point[0], current_point[1]) < 5:
                print(f"Point {self.path_index + 1} atteint.")  # Afficher le message
                self.path_index += 1  # Passer au point suivant
                if self.path_index >= len(self.path) :
                    print("L'agent a atteint l'objectif !")
                    return

        self.draw_agent_and_goal()
        self.draw_path(self.path)

        self.root.after(10, self.update)


    def draw_path(self, path):
        for (x, y) in path:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="purple")

    def start(self):
        self.update()
        self.root.mainloop()

if __name__ == "__main__":
    env = Environment(num_obstacles=5, grid_resolution=5)  # Résolution fine pour plus de points
    env.start()