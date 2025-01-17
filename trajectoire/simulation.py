import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import math
from tkinter import Canvas
import math

import math

# Variable de contrôle pour l'arrêt
running = False
move_job = None  # Ce sera l'identifiant de l'appel after

def simulation_drone(vehicle, canvas, finish_pos):
    global running, move_job

    def move():
        global running
        nonlocal vehicle

        if not running:
            return  # Arrêter si running est faux

        # Calculer la direction vers la cible
        dx = finish_pos[0] - vehicle.x
        dy = finish_pos[1] - vehicle.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > 10:  # Vérifier si le drone n'est pas encore arrivé
            # Calculer l'orientation nécessaire pour se diriger vers la cible
            target_orientation = math.atan2(dy, dx)
            angle_diff = target_orientation - vehicle.orientation

            # Ajuster l'orientation graduellement
            if abs(angle_diff) > 0.1:
                vehicle.turn(angle_diff * 0.1)

            # Déplacer le drone
            vehicle.speed = 2
            vehicle.move(1)

            # Mettre à jour la position sans recréer un nouveau rectangle
            canvas.coords(
                vehicle.drone_id,
                vehicle.x - 20, vehicle.y - 10,
                vehicle.x + 20, vehicle.y + 10,
            )

            # Reprogrammer pour continuer le déplacement
            move_job = canvas.after(50, move)
        else:
            print("Drone arrivé à destination!")
            running = False  # Arrêter la simulation

    def log_position():
        if not running:
            return
        print(f"Drone position: x={vehicle.x}, y={vehicle.y}, orientation={vehicle.orientation} target={finish_pos}")
        canvas.after(5000, log_position)

    move()
    log_position()


def start() :
    global running
    running = True


def stop():
    global running, move_job

    if running:
        print("Simulation arrêtée!")
        running = False  # Désactiver la simulation
