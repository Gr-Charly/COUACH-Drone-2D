import math

class Drone:
    def __init__(self, x=0, y=0, orientation=0, speed=0):
        self.x = x  # Position en x
        self.y = y  # Position en y
        self.orientation = orientation  # Orientation en radians
        self.speed = speed  # Vitesse

    def move(self, time_step):
        """Déplace le véhicule en fonction de sa vitesse et de son orientation."""
        self.x += self.speed * math.cos(self.orientation) * time_step
        self.y += self.speed * math.sin(self.orientation) * time_step

    def turn(self, angle):
        """Tourne le véhicule."""
        self.orientation += angle
