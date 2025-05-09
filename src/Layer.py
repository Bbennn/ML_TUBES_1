from .Regularization import Regularization
from .Activation import *
from .Initialization import *
from .RMSNorm import RMSNorm
from typing import Tuple


class Layer:
    activation: ActivationFunction
    initializer: InitializationFunction
    regularizer: Regularization
    alpha_regularizer: float
    weight: np.ndarray
    bias: np.ndarray
    weight_gradient: np.ndarray  # pakai weight gradient karena perlu diplot nanti
    bias_gradient: np.ndarray
    input: np.ndarray
    linear_combinations: np.ndarray

    def __init__(
        self,
        shape: Tuple[int, int],
        activation: ActivationFunction,
        initializer: InitializationFunction,
        regularizer: Regularization,
        alpha_regularizer: float,
        use_rms_norm: bool = False,
    ):
        self.activation = activation
        self.initializer = initializer
        self.weight = initializer.init(shape).astype(np.float64)
        self.bias = np.zeros(shape[1])
        self.regularizer = regularizer
        self.alpha_regularizer = alpha_regularizer
        self.rms_norm = (
            RMSNorm(shape[1]) if use_rms_norm else None
        )  # Apply RMSNorm if enabled

    def run(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.clip(inputs, -1e3, 1e3)
        self.input = inputs
        self.linear_combinations = np.matmul(inputs, self.weight) + self.bias

        if self.rms_norm:
            self.linear_combinations = self.rms_norm(
                self.linear_combinations
            )  # Apply RMSNorm

        outputs = self.activation.fn(self.linear_combinations)
        return outputs

    def update_gradient(self, delta: np.ndarray, ancestor: "Layer"):
        delta = delta.dot(ancestor.weight.T)
        delta = delta * self.activation.do_ds(self.linear_combinations)
        self.weight_gradient = np.dot(self.input.T, delta)
        self.bias_gradient = np.sum(delta, axis=0)
        if self.regularizer != None:
            self.weight_gradient = (
                self.weight_gradient
                + self.alpha_regularizer * self.regularizer.dR_dw(self.weight)
            )
            self.bias_gradient = (
                self.bias_gradient
                + self.alpha_regularizer * self.regularizer.dR_dw(self.bias)
            )

        return delta

    def update_weight(self, learning_rate: float):
        self.weight -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient
