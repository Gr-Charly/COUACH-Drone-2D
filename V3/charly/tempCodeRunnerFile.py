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