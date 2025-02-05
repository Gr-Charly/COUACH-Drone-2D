import tkinter as tk
import numpy as np

# Fonction pour générer des valeurs aléatoires pour les poids
def init_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output

# Fonction d'activation (sigmoïde)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Réseau de neurones de base à 3 couches
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden, self.weights_hidden_output = init_weights(input_size, hidden_size, output_size)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.output = sigmoid(self.output_input)
        return self.output

# Classe Tkinter pour afficher le réseau
class NeuralNetworkVisualizer:
    def __init__(self, root, nn):
        self.root = root
        self.nn = nn
        self.canvas = tk.Canvas(self.root, width=800, height=500, bg="white")  # Fenêtre plus grande
        self.canvas.pack()
        self.draw_network()

    def draw_network(self):
        # Nouveau calcul des espacements pour chaque couche
        layer_width = 150
        layer_height = 120
        neuron_radius = 30
        
        # Affichage des neurones de la couche d'entrée
        for i in range(self.nn.input_size):
            self.canvas.create_oval(100, 50 + i * layer_height, 100 + neuron_radius * 2, 50 + (i + 1) * layer_height, fill="blue")
            self.canvas.create_text(100 + neuron_radius, 50 + (i + 0.5) * layer_height, text=f'Input {i+1}', fill="white")
        
        # Affichage des neurones de la couche cachée
        for i in range(self.nn.hidden_size):
            self.canvas.create_oval(300, 50 + i * layer_height, 300 + neuron_radius * 2, 50 + (i + 1) * layer_height, fill="green")
            self.canvas.create_text(300 + neuron_radius, 50 + (i + 0.5) * layer_height, text=f'Hidden {i+1}', fill="white")
        
        # Affichage des neurones de la couche de sortie
        for i in range(self.nn.output_size):
            self.canvas.create_oval(500, 50 + i * layer_height, 500 + neuron_radius * 2, 50 + (i + 1) * layer_height, fill="red")
            self.canvas.create_text(500 + neuron_radius, 50 + (i + 0.5) * layer_height, text=f'Output {i+1}', fill="white")

        # Dessiner les connexions entre les couches
        self.draw_connections()

    def draw_connections(self):
        layer_height = 120
        # Connexions entre la couche d'entrée et la couche cachée
        for i in range(self.nn.input_size):
            for j in range(self.nn.hidden_size):
                self.canvas.create_line(100 + 30, 50 + i * layer_height + 30, 300 + 30, 50 + j * layer_height + 30, width=2, fill="black")
        
        # Connexions entre la couche cachée et la couche de sortie
        for i in range(self.nn.hidden_size):
            for j in range(self.nn.output_size):
                self.canvas.create_line(300 + 30, 50 + i * layer_height + 30, 500 + 30, 50 + j * layer_height + 30, width=2, fill="black")

# Fonction principale pour démarrer l'application
def main():
    input_size = 3   # Nombre de neurones dans la couche d'entrée
    hidden_size = 4  # Nombre de neurones dans la couche cachée
    output_size = 2  # Nombre de neurones dans la couche de sortie
    
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Création de la fenêtre Tkinter
    root = tk.Tk()
    root.title("Visualisation du Réseau de Neurones")
    
    # Création du visualiseur du réseau
    visualizer = NeuralNetworkVisualizer(root, nn)

    # Lancement de l'application Tkinter
    root.mainloop()

if __name__ == "__main__":
    main()
