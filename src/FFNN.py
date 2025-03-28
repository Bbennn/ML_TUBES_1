import numpy as np
import pickle
from typing import List, Literal

from Activation import get_activation, ActivationFunction, SoftmaxActivationFunction
from Initialization import get_initializer, InitializationFunction
from Loss import get_loss_function, ErrorFunction
from Layer import Layer
import matplotlib.pyplot as plt
import networkx as nx

class FFNN:
    activations: List[str]
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
        self.activations = []
        for i in range(len(activations)):
            activation_func = get_activation(activations[i]) if isinstance(activations[i], str) else activations[i]
            self.layers.append(Layer((layer_sizes[i], layer_sizes[i + 1]), activation_func, self.weight_initializer))
            self.activations.append(activation_func.name())
    
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
        if (self.activations[-1] == "softmax"):
            delta = delta * self.layers[-1].activation.do_ds(self.layers[-1].linear_combinations, delta)
        else:
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
    
    # def show_graph(self):
    #     G = nx.DiGraph()
    #     positions = {}
    #     node_colors = []
    #     node_list = [] 

    #     for i in range(len(self.layers_sizes)):
    #         if i == 0:
    #             prefix = "x"
    #             color = "white"
    #         elif i == len(self.layers_sizes) - 1:
    #             prefix = "y"
    #             color = "orange"
    #         else:
    #             prefix = f"h{i}"
    #             color = "brown"
            
    #         node_name = f"{prefix}({self.layers_sizes[i]})"
    #         positions[node_name] = (i, 0)
    #         node_colors.append(color)
    #         node_list.append(node_name)

    #     edges = [(node_list[i], node_list[i+1]) for i in range(len(node_list)-1)]

    #     G.add_edges_from(edges)

    #     plt.figure(figsize=(3*len(self.layers_sizes), 3))

    #     nx.draw(G, pos=positions, with_labels=True, node_size=3000,
    #             node_color=node_colors, edge_color="blue",
    #             font_size=12, font_weight="bold", edgecolors="black")

    #     edge_labels = {}
    #     matrix_legend = []
    #     # edge_labels = {edge: f"W_{edge[0]}{edge[1]}" for edge in edges}
    #     for i in range(len(edges)):
    #         if i == 0:
    #             label = f"W_{edges[i][0][:1]}{edges[i][1][:2]}\n"
    #         elif i == len(edges) - 1:
    #             label = f"W_{edges[i][0][:2]}{edges[i][1][:1]}\n"
    #         else:
    #             label = f"W_{edges[i][0][:2]}{edges[i][1][:2]}\n"
            
    #         edge_labels[edges[i]] = label
    #         matrix_str = np.array2string(self.layers[i].weight, precision=2, separator=', ')    
    #         matrix_legend.append(f"{label} = {matrix_str}")  


    #     nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=8)

    #     plt.title("Compact Style Graph", fontsize=14)
    #     # Legend
    #     plt.subplot(2, 1, 2)  # Bottom subplot for legends
    #     plt.axis('off')

        
    #     # Create a single string for all matrix legends
    #     full_legend_text = "\n\n".join(matrix_legend)
        
    #     plt.text(-0.05, 0.5, full_legend_text, 
    #             fontsize=8, 
    #             verticalalignment='top', 
    #             horizontalalignment='left', 
    #             )
        
    #     plt.tight_layout()
    #     plt.show()
    def show_graph(self):

        # Create directed graph
        G = nx.DiGraph()
        positions = {}
        node_colors = []
        node_list = []
        
        # Create nodes for each layer
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
        
        # Create edges between layers
        edges = [(node_list[i], node_list[i+1]) for i in range(len(node_list)-1)]
        G.add_edges_from(edges)
        
        # Create figure with appropriate size
        plt.figure(figsize=(3*len(self.layers_sizes), 10))  # Increased height to accommodate more legends
        
        # Create a subplot for the graph
        plt.subplot(2, 1, 1)  # Top subplot for the graph
        
        # Draw the graph
        nx.draw(G, pos=positions, with_labels=True, node_size=3000,
                node_color=node_colors, edge_color="blue",
                font_size=12, font_weight="bold", edgecolors="black")
        
        # Prepare edge labels and matrix legends
        edge_labels = {}
        weight_legend = []
        nabla_weight_legend = []
        
        for i in range(len(edges)):
            if i == 0:
                label = f"W_{edges[i][0][:1]}{edges[i][1][:2]}"
                nabla_label = f"∇W_{edges[i][0][:1]}{edges[i][1][:2]}"
            elif i == len(edges) - 1:
                label = f"W_{edges[i][0][:2]}{edges[i][1][:1]}"
                nabla_label = f"∇W_{edges[i][0][:2]}{edges[i][1][:1]}"
            else:
                label = f"W_{edges[i][0][:2]}{edges[i][1][:2]}"
                nabla_label = f"∇W_{edges[i][0][:2]}{edges[i][1][:2]}"
            
            edge_labels[edges[i]] = label
            
            # Convert weight matrix to a more readable string
            matrix_str = np.array2string(self.layers[i].weight, 
                                        precision=2, 
                                        separator=', ', 
                                        suppress_small=True)
            weight_legend.append(f"{label} = {matrix_str}")
            
            # Convert nabla weight matrix to a string
            nabla_matrix_str = np.array2string(self.layers[i].weight_gradient, 
                                                precision=2, 
                                                separator=', ', 
                                                suppress_small=True)
            nabla_weight_legend.append(f"{nabla_label} = {nabla_matrix_str}")
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=8)
        
        plt.title("Compact Style Graph", fontsize=14)

        # Create a subplot for legends
        plt.subplot(2, 1, 2)  # Bottom subplot for legends
        plt.axis('off')  # Turn off axis
        
        # Combine legends
        full_legend_text_1 = "Weight Matrices:\n" + "\n\n".join(weight_legend)
        full_legend_text_2 = "\n\nNabla Weight Matrices:\n" + "\n\n".join(nabla_weight_legend)
        
        # Add legends with left alignment
        plt.text(0.1, 1.4, full_legend_text_1, 
                fontsize=8, 
                verticalalignment='top', 
                horizontalalignment='left', 
                family='monospace')
                # transform=plt.gca().transAxes)  # Use axes coordinates
        plt.text(0.6, 1.4, full_legend_text_2, 
                fontsize=8, 
                verticalalignment='top', 
                horizontalalignment='left', 
                family='monospace')
                # transform=plt.gca().transAxes)  # Use axes coordinates
        
        plt.tight_layout()
        plt.show()
    @staticmethod
    def load_model(filename: str) -> "FFNN":
        """Loads the model from a file."""
        with open(filename, 'rb') as file:
            return pickle.load(file)
