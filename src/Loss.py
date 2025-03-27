import numpy as np

class ErrorFunction:
    def fn(self, X): ...
    def dE_do(self, T, X): ...

class MeanSquaredError(ErrorFunction):
    def fn(self, T, O):
        return (1 / T.shape[0]) * np.sum((T - O) ** 2)

    def dE_do(self, T, O):
        return (-2 / T.shape[0]) * (T - O)
    
class BinaryCrossEntropy(ErrorFunction):
    def fn(self, T, O):
        O = np.clip(O, 1e-15, 1 - 1e-15) 
        return -np.mean(T * np.log(O) + (1 - T) * np.log(1 - O))

    def dE_do(self, T, O):
        O = np.clip(O, 1e-15, 1 - 1e-15) 
        return (O - T) / (O * (1 - O) + 1e-15) / T.shape[0]

class CategoricalCrossEntropy(ErrorFunction):
    def fn(self, T, O):
        eps = 1e-15
        O = np.clip(O, eps, 1.0)
        
        if len(T.shape) == 2:
            return -np.mean(np.sum(T * np.log(O), axis=1))
        else:
            n = T.shape[0]
            return -np.sum(np.log(O[np.arange(n), T])) / n
    
    def dE_do(self, T, O):
        O = np.clip(O, 1e-15, 1)  
        return -T / (O + 1e-15) / T.shape[0]
