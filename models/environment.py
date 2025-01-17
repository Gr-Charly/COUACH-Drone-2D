# models/environment.py

def get_obstacles(carte):
    """Retourne les obstacles en fonction de la carte."""
    if carte == "carte1":
        obstacles = [
            (100, 100, 150, 150),  # Premier obstacle
            (300, 200, 350, 250),  # Deuxième obstacle
            (500, 50, 550, 100)    # Troisième obstacle
        ]
        return obstacles, (50, 50), (760, 360)
    elif carte == "carte2":
        obstacles = [
            (200, 150, 250, 200),  # Exemple d'obstacle carte 2
            (400, 100, 450, 150)
        ]
        return obstacles, (100, 200), (760, 360)
    elif carte == "carte3":
        obstacles = [
            (50, 50, 100, 100),    # Exemple d'obstacle carte 3
            (600, 300, 650, 350)
        ]
        return obstacles, (760, 360), (20, 20)
    else:
        return []
