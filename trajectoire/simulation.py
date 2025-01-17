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
        global running  # Déclare 'running' comme globale pour pouvoir y accéder
        nonlocal vehicle

        if not running:
            return  # Si la simulation est arrêtée, on ne continue pas

        # Calculer la direction vers la cible
        dx = finish_pos[0] - vehicle.x
        dy = finish_pos[1] - vehicle.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > 1:  # Vérifier si le drone n'est pas encore arrivé
            # Calculer l'orientation nécessaire pour se diriger vers la cible
            target_orientation = math.atan2(dy, dx)
            angle_diff = target_orientation - vehicle.orientation

            # Ajuster l'orientation graduellement (limitation pour éviter des changements brusques)
            if abs(angle_diff) > 0.1:
                vehicle.turn(angle_diff * 0.1)

            # Déplacer le drone
            vehicle.speed = 2  # Régler une vitesse constante
            vehicle.move(1)  # Utiliser un pas de temps de 1

            # Redessiner le drone
            canvas.delete(vehicle.drone_id)  # Supprimer l'ancien dessin
            vehicle.drone_id = canvas.create_rectangle(
                vehicle.x - 20, vehicle.y - 10, vehicle.x + 20, vehicle.y + 10,
                fill='black'
            )

            # Reprogrammer pour continuer le déplacement
            move_job = canvas.after(50, move)  # Appel récursif pour continuer le déplacement
        else:
            print("Drone arrivé à destination!")
            running = False  # Arrêter la simulation une fois la destination atteinte

    def log_position():
        if not running:
            return  # Si la simulation est arrêtée, ne pas continuer à loguer
        # Afficher la position du drone et la position cible toutes les 5 secondes
        print(f"Drone position: x={vehicle.x}, y={vehicle.y}, target={finish_pos}")
        # Planifier la prochaine exécution après 5 secondes
        canvas.after(5000, log_position)

    move()  # Lancer le déplacement du drone
    log_position()  # Lancer la fonction de log

def start() :
    global running
    running = True


def stop():
    global running, move_job

    if running:
        print("Simulation arrêtée!")
        running = False  # Désactiver la simulation
