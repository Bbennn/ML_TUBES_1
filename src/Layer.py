from Activation import *
from Initialization import *
from typing import Tuple, List
class Layer:
    shape: Tuple[int,int]
    activation: ActivationFunction
    initializer: InitializationFunction
    weight: np.ndarray
    bias: List[float]
    gradient_weight: np.ndarray
    gradient_bias: np.ndarray
    input: np.ndarray
    def __init__(self, shape: Tuple[int,int], activation: ActivationFunction, initializer: InitializationFunction):
        self.shape = shape
        self.activation = activation
        self.initializer = initializer
        self.weight = initializer.init(shape=shape)
        self.bias = np.zeros(shape[1])
    def info(self):
        print(self.activation)
        print(self.initializer)
        print(self.weight)
        print(self.bias)
        print(self.gradient_weight)
        print(self.gradient_bias)
        print(self.input)
s = (5,10)
l1 = Layer(shape=s, activation=SigmoidActivationFunction, initializer=ZeroInitialization())
l1.info()