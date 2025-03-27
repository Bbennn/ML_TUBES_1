import numpy as np
import pickle
from typing import List, Literal

from .Activation import get_activation, ActivationFunction
from .Initialization import get_initializer, InitializationFunction
from .Loss import get_loss_function, ErrorFunction
from .Layer import Layer

class FFNN:
    activations: List[ActivationFunction]
    loss: ErrorFunction
    weight_initializer: InitializationFunction
    layers: List[Layer]
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
            delta = self.layers[i].update_gradient(delta, self.layers[i-1] if i > 0 else None)
    
    def update_weights(self, learning_rate: float):
        """Updates weights using the computed gradients."""
        for layer in self.layers:
            layer.update_weight(learning_rate)
    
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, epochs: int, learning_rate: float, batch_size: int, verbose: bool = False):
        """Trains the neural network using mini-batch gradient descent."""
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch, Y_batch = X_train[indices[start_idx:end_idx]], Y_train[indices[start_idx:end_idx]]
                
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
    
    @staticmethod
    def load_model(filename: str) -> "FFNN":
        """Loads the model from a file."""
        with open(filename, 'rb') as file:
            return pickle.load(file)
