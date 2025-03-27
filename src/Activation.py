import numpy as np
class ActivationFunction:
    def fn(X): ...
    def do_ds(nets): ...
    def name(): ... 


class LinearActivationFunction(ActivationFunction):
    def fn(N):
        return N

    def do_ds(N):
        return np.ones_like(N)
    
    def name(): return "linear"


class ReluActivationFunction(ActivationFunction):
    def fn(N):
        return np.maximum(0, N)

    def do_ds(N):
        return np.where(N > 0, 1, 0)
    
    def name(): return "relu"


class SigmoidActivationFunction(ActivationFunction):
    def fn(N):
        N = np.clip(N, -500, 500)
        return 1 / (1 + np.exp(-N))

    def do_ds(N):
        O = SigmoidActivationFunction.fn(N)
        return O * (np.ones(O.shape) - O)
    
    def name(): return "sigmoid"


class TanhActivationFunction(ActivationFunction):
    def fn(N):
        return np.tanh(N)

    def do_ds(N):
        return 1 - np.tanh(N) ** 2
    
    def name(): return "tanh"

class SoftmaxActivationFunction(ActivationFunction):
    def fn(N):
        N = np.clip(N, -500, 500)  # Prevent overflow
        exp_N = np.exp(N - np.max(N, axis=-1, keepdims=True))  # Numerical stability
        return exp_N / np.sum(exp_N, axis=-1, keepdims=True)

    def do_ds(N, dE_do):
        """
        Compute the Jacobian-vector product of the softmax function.
        Instead of returning a 3D Jacobian, it returns a 2D matrix as required.
        """
        S = SoftmaxActivationFunction.fn(N)
        batch_size, _ = S.shape
        jacobian_vector_product = np.zeros_like(S)

        for i in range(batch_size):
            s_i = S[i].reshape(-1, 1)  # Convert to column vector
            jacobian_i = np.diagflat(s_i) - np.dot(s_i, s_i.T)  # Compute the Jacobian
            jacobian_vector_product[i] = np.dot(jacobian_i, dE_do[i])  # Multiply by dE/do

        return (-1) * jacobian_vector_product
    
    def name(): return "softmax"


ACTIVATION_FUNCTIONS = {
    "linear": LinearActivationFunction,
    "relu": ReluActivationFunction,
    "sigmoid": SigmoidActivationFunction,
    "tanh": TanhActivationFunction,
    "softmax": SoftmaxActivationFunction
}

def get_activation(act_name: str):
    name = act_name.lower()
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Activation Function `{name}` not available")
    return ACTIVATION_FUNCTIONS[name]