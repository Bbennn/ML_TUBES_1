from Activation import *
from Initialization import *
from typing import Tuple, List
class Layer:
    activation: ActivationFunction
    initializer: InitializationFunction
    weight: np.ndarray
    bias: np.ndarray
    weight_gradient: np.ndarray         # pakai weight gradient karena perlu diplot nanti
    bias_gradient: np.ndarray
    input: np.ndarray
    linear_combinations: np.ndarray

    def __init__(self, shape: Tuple[int,int], activation: ActivationFunction, initializer: InitializationFunction):
        self.activation = activation
        self.initializer = initializer
        self.weight = initializer.init(shape)
        self.bias = np.zeros(shape[1])
    
    def run(self, inputs: np.ndarray) -> np.ndarray:
        self.input = inputs
        self.linear_combinations = np.matmul(inputs, self.weight) + self.bias
        outputs = self.activation.fn(self.linear_combinations)
        return outputs

    def update_gradient(self, delta: np.ndarray, ancestor: "Layer"):
        delta = delta.dot(ancestor.weight.T)
        delta = delta * self.activation.do_ds(self.linear_combinations)
        batch_size = self.input.shape[0]
        self.weight_gradient = np.dot(self.input.T, delta) / batch_size
        self.bias_gradient = np.mean(delta, axis=0)
        return delta
    
    def update_weight(self, learning_rate: float):
        self.weight -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient

    def info(self):
        print(self.activation)
        print(self.initializer)
        print(self.weight)
        print(self.bias)
        print(self.input)