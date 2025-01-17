import subprocess
from tkinter import *
from models.drone import Drone
from models.environment import get_obstacles
from trajectoire.simulation import simulation_drone, stop, start

ENV = 0
START = 0
STOP = 0

def draw_vehicle(x, y, width=30, height=20):
    """Dessine un rectangle représentant le drone."""
    return canvas.create_rectangle(
        x - width // 2, y - height // 2,  # Coin supérieur gauche
        x + width // 2, y + height // 2,  # Coin inférieur droit
        fill='black'
    )

__all__ = ['canvas', 'draw_vehicle', 'Drone']
obstacle_ids = []
drone_id = None
finish_id = None  # ID pour le carré "fin"

def env(carte):
    global ENV, obstacle_ids, drone_id, finish_id, vehicle
    ENV = 1

    # Nettoyer les éléments existants
    for obs_id in obstacle_ids:
        canvas.delete(obs_id)
    obstacle_ids.clear()

    if drone_id is not None:
        canvas.delete(drone_id)
    if finish_id is not None:
        canvas.delete(finish_id)

    # Obtenir les données de la carte
    obstacles, position_init, position_fin = get_obstacles(carte)
    x_init, y_init = position_init
    x_fin, y_fin = position_fin

    # Initialiser le drone
    vehicle = Drone(x=x_init, y=y_init, speed=2)  # Vitesse de 2
    drone_id = canvas.create_rectangle(
        vehicle.x - 20, vehicle.y - 10, vehicle.x + 20, vehicle.y + 10, fill='black'
    )
    vehicle.drone_id = drone_id  # Associer l'ID au drone

    # Dessiner les obstacles
    for obs in obstacles:
        obs_id = canvas.create_rectangle(obs[0], obs[1], obs[2], obs[3], fill='green')
        obstacle_ids.append(obs_id)

    # Dessiner la position finale
    finish_size = 30
    finish_id = canvas.create_rectangle(
        x_fin - finish_size // 2, y_fin - finish_size // 2,
        x_fin + finish_size // 2, y_fin + finish_size // 2,
        fill='red'
    )




def update_vehicle(vehicle, vehicle_id):
    global drone_id

    # Supprimer l'ancien dessin du drone
    canvas.delete(vehicle_id)

    # Redessiner le drone et mettre à jour l'ID
    drone_id = draw_vehicle(vehicle.x, vehicle.y, width=40, height=20)

    # Continuer à mettre à jour si le drone est dans les limites
    if 0 <= vehicle.x <= 800 and 0 <= vehicle.y <= 400:
        canvas.after(100, update_vehicle, vehicle, drone_id)


def start_simulation():
    global ENV, STOP, START, vehicle, finish_id

    if ENV == 1 and START == 0:  # Empêche de recréer un drone si la simulation est déjà lancée
        print("Simulation démarrée!")

        # Récupérer la position centrale du carré "finish"
        coords = canvas.coords(finish_id)
        finish_pos = ((coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2)

        # Assurez-vous que le drone n'est pas redessiné ici
        start()
        simulation_drone(vehicle, canvas, finish_pos)

        START = 1
        STOP = 0




def stop_simulation():
    global ENV, STOP, START
    if(START == 1) :
        print("Simulation arrêtée!")
        stop()
        STOP = 1
        START = 0

if __name__ == "__main__":
    # Création de la fenêtre principale
    fenetre = Tk()
    fenetre.geometry('800x400')
    fenetre.title('Simulation 2D - Drone COUACH')
    fenetre['bg'] = 'blue'
    fenetre.resizable(height=True, width=True)

    # Création du canvas
    canvas = Canvas(fenetre, width=800, height=400, bg='blue')
    canvas.pack()

    # Création du menu
    menu_2D = Menu(fenetre)

    # Sous onglet cartes
    cartes = Menu(menu_2D, tearoff=0)
    cartes.add_command(label="Carte1", command=lambda: env("carte1"))
    cartes.add_command(label="Carte2", command=lambda: env("carte2"))
    cartes.add_command(label="Carte3", command=lambda: env("carte3"))

    # Nos onglets 
    menu_2D.add_cascade(label="Cartes", menu=cartes)
    menu_2D.add_cascade(label="Start", command=start_simulation)
    menu_2D.add_cascade(label="Stop", command=stop_simulation)

    fenetre.config(menu=menu_2D)

    fenetre.mainloop()
