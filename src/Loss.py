import numpy as np

class ErrorFunction:
    def fn(T, O): ...
    def dE_do(T, O): ...

class MeanSquaredError(ErrorFunction):
    def fn(T, O):
        return (1 / T.shape[0]) * np.sum((T - O) ** 2)

    def dE_do(T, O):
        return (-2 / T.shape[0]) * (T - O)
    
class BinaryCrossEntropy(ErrorFunction):
    def fn(T, O):
        O = np.clip(O, 1e-15, 1 - 1e-15) 
        return -np.mean(T * np.log(O) + (1 - T) * np.log(1 - O))

    def dE_do(T, O):
        O = np.clip(O, 1e-15, 1 - 1e-15) 
        return (O - T) / (O * (1 - O) + 1e-15) / T.shape[0]

class CategoricalCrossEntropy(ErrorFunction):
    def fn(T, O):
        eps = 1e-15
        O = np.clip(O, eps, 1.0)
        
        if len(T.shape) == 2:
            return -np.mean(np.sum(T * np.log(O), axis=1))
        else:
            n = T.shape[0]
            return -np.sum(np.log(O[np.arange(n), T])) / n
    
    def dE_do(T, O):
        O = np.clip(O, 1e-15, 1)  
        return -T / (O + 1e-15) / T.shape[0]

LOSS_FUNCTION = {
    "mse": MeanSquaredError,
    "bce": BinaryCrossEntropy,
    "cce": CategoricalCrossEntropy
}

def get_loss_function(loss_name: str):
    name = loss_name.lower()
    if name not in LOSS_FUNCTION:
        raise ValueError(f"Loss Function `{name}` not available")
    return LOSS_FUNCTION[name]