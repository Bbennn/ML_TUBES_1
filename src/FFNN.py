import numpy as np
import pickle
from typing import List, Literal

from Activation import get_activation, ActivationFunction
from Initialization import get_initializer, InitializationFunction
from Loss import get_loss_function, ErrorFunction
from Layer import Layer
import matplotlib.pyplot as plt
import networkx as nx

class FFNN:
    activations: List[ActivationFunction]
    loss: ErrorFunction
    weight_initializer: InitializationFunction
    layers: List[Layer]
    layers_sizes: List[int]
    def __init__(self,
                 layer_sizes: List[int],
                 activations: List[Literal["linear", "relu", "sigmoid", "tanh", "softmax"] | ActivationFunction],
                 loss: Literal["mse", "bce", "cce"] | ErrorFunction = "mse",
                 weight_initializer: Literal["zero", "uniform", "normal", "xavier", "he"] | InitializationFunction = "uniform",
                 weight_init_args=None):
        """
        layer_sizes: list of neurons per layer (including input and output layers)
        activations: activation functions for each layer except input
        loss: loss function name or instance
        weight_initializer: weight initialization method or instance
        """
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least 1 layer (output)")
        
        if len(layer_sizes) - 1 != len(activations):
            raise ValueError("Number of activations must be one less than number of layers")
        
        self.n_layers = len(layer_sizes) - 1
        self.layers_sizes = layer_sizes

        if weight_init_args is None:
            weight_init_args = {}
        
        if isinstance(weight_initializer, str):
            self.weight_initializer = get_initializer(weight_initializer, **weight_init_args)
        else:
            self.weight_initializer = weight_initializer
        
        if isinstance(loss, str):
            self.loss = get_loss_function(loss)
        else:
            self.loss = loss
        self.layers = []
        for i in range(len(activations)):
            activation_func = get_activation(activations[i]) if isinstance(activations[i], str) else activations[i]
            self.layers.append(Layer((layer_sizes[i], layer_sizes[i + 1]), activation_func, self.weight_initializer))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs forward propagation."""
        for layer in self.layers:
            X = layer.run(X)
        return X
    
    def backward(self, X: np.ndarray, Y: np.ndarray):
        """Performs backward propagation and computes gradients."""
        O = self.forward(X)
        
        # Compute loss derivative for output layer
        delta = self.loss.dE_do(Y, O)
        delta = delta * self.layers[-1].activation.do_ds(self.layers[-1].linear_combinations)
        self.layers[-1].weight_gradient = np.dot(self.layers[-1].input.T, delta) / self.layers[-1].input.shape[0]
        self.layers[-1].bias_gradient = np.mean(delta, axis=0)
        
        # Compute weight gradients and bias gradients
        for i in range(len(self.layers) - 2, -1, -1):
            delta = self.layers[i].update_gradient(delta, self.layers[i+1])
    
    def update_weights(self, learning_rate: float):
        """Updates weights using the computed gradients."""
        for layer in self.layers:
            layer.update_weight(learning_rate)
    
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, epochs: int, learning_rate: float, batch_size: int, verbose: bool = False):
        """Trains the neural network using mini-batch gradient descent."""
        X_train = np.array(X_train)  # Ensure NumPy array
        Y_train = np.array(Y_train)  # Ensure NumPy array
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                X_batch, Y_batch = X_train[batch_indices], Y_train[batch_indices]  # Use batch_indices

                self.backward(X_batch, Y_batch)
                self.update_weights(learning_rate)

            if verbose and epoch % (epochs // 10) == 0:
                loss_value = self.loss.fn(Y_train, self.forward(X_train))
                print(f"Epoch {epoch}/{epochs}, Loss: {loss_value:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the model's predictions."""
        return self.forward(X)
    
    def save_model(self, filename: str):
        """Saves the model to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def plot_weight(self, layers: List[int]):
        for idx in layers:
            plt.figure()  
            data = self.layers[idx].weight.flatten()
            plt.hist(data, bins=100)
            plt.title(f"Layer {idx} Weight Distribution")
            plt.xlabel("Weight Values")
            plt.ylabel("Frequency")
            plt.show() 
            
    def plot_gradient_weight(self, layers: List[int]):
        for idx in layers:
            plt.figure()  
            data = self.layers[idx].weight_gradient.flatten()
            plt.hist(data, bins=100)
            plt.title(f"Layer {idx} Weight Distribution")
            plt.xlabel("Weight Values")
            plt.ylabel("Frequency")
            plt.show() 
    
    def show_graph(self):
        G = nx.DiGraph()
        positions = {}
        node_colors = []
        node_list = [] 

        for i in range(len(self.layers_sizes)):
            if i == 0:
                prefix = "x"
                color = "white"
            elif i == len(self.layers_sizes) - 1:
                prefix = "y"
                color = "orange"
            else:
                prefix = f"h{i}"
                color = "brown"
            
            node_name = f"{prefix}({self.layers_sizes[i]})"
            positions[node_name] = (i, 0)
            node_colors.append(color)
            node_list.append(node_name)

        edges = [(node_list[i], node_list[i+1]) for i in range(len(node_list)-1)]

        G.add_edges_from(edges)

        plt.figure(figsize=(3*len(self.layers_sizes), 3))
        nx.draw(G, pos=positions, with_labels=True, node_size=3000,
                node_color=node_colors, edge_color="blue",
                font_size=12, font_weight="bold", edgecolors="black")

        edge_labels = {}
        matrix_legend = []
        # edge_labels = {edge: f"W_{edge[0]}{edge[1]}" for edge in edges}
        for i in range(len(edges)):
            if i == 0:
                label = f"W_{edges[i][0][:1]}{edges[i][1][:2]}"
            elif i == len(edges) - 1:
                label = f"W_{edges[i][0][:2]}{edges[i][1][:1]}"
            else:
                label = f"W_{edges[i][0][:2]}{edges[i][1][:2]}"
            
            edge_labels[edges[i]] = label
            matrix_str = np.array2string(self.layers[i].weight, precision=2, separator=', ')
            matrix_legend.append(f"{label} = {matrix_str}")  


        nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=8)

        # Add legend text below the graph
        y_min = min(y for _, y in positions.values()) - 0.3  # Reduce gap
        x_offset = 0  # Move closer to the graph
        line_spacing = 0.2  # Reduce space between legend items

        for i, text in enumerate(matrix_legend):
            plt.text(x_offset, y_min - (i * line_spacing), text, 
                     fontsize=10, ha='left', va='top', color='black')

        plt.title("Compact Style Graph", fontsize=14)
        plt.show()

    @staticmethod
    def load_model(filename: str) -> "FFNN":
        """Loads the model from a file."""
        with open(filename, 'rb') as file:
            return pickle.load(file)

# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split

# train_samples = 1000

# X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# def expand(i):
#     res = [0 for _ in range(0, 10)]
#     res[i] = 1 
#     return res

# y = [expand(int(v)) for v in y]
# # X = np.float128(X)
# # print(y)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=train_samples, test_size=10000
# )


# layer_size = [784, 24, 24, 24, 10, 10]
# activations = ["sigmoid", "sigmoid", "sigmoid", "relu", "tanh"]

# model = FFNN(layer_sizes=layer_size, activations=activations, loss="mse", weight_initializer="normal", weight_init_args={"seed": 73})
# model.fit(X_train, y_train, 10, 0.1, 10, True)