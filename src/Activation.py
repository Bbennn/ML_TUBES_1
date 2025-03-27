import numpy as np
class ActivationFunction:
    def fn(X): ...
    def do_ds(nets): ...


class LinearActivationFunction(ActivationFunction):
    def fn(N):
        return N

    def do_ds(O):
        return np.ones_like(O)


class ReluActivationFunction(ActivationFunction):
    def fn(N):
        return np.maximum(0, N)

    def do_ds(O):
        return np.where(O > 0, 1, 0)


class SigmoidActivationFunction(ActivationFunction):
    def fn(N):
        N = np.clip(N, -500, 500)
        return 1 / (1 + np.exp(-N))

    def do_ds(O):
        return O * (np.ones(O.shape) - O)


class TanhActivationFunction(ActivationFunction):
    def fn(N):
        return np.tanh(N)

    def do_ds(O):
        return 1 - np.tanh(O) ** 2

class SoftmaxActivationFunction(ActivationFunction):
    def fn(N):
        N = np.clip(N, -500, 500)
        exp_N = np.exp(N - np.max(N, axis=-1, keepdims=True))
        return exp_N / np.sum(exp_N, axis=-1, keepdims=True)

    def do_ds(O):
        batch_size, num_classes = O.shape
        jacobian = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            softmax_i = O[i].reshape(-1, 1)  # Convert to column vector
            jacobian[i] = np.diagflat(softmax_i) - np.dot(softmax_i, softmax_i.T)

        return jacobian

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