import numpy as np

class RMSNorm:
    def __init__(self, shape, epsilon=1e-8):
        """
        Root Mean Square Layer Normalization.
        
        Args:
            shape: Tuple representing the number of neurons in the layer.
            epsilon: Small constant to avoid division by zero.
        """
        self.epsilon = epsilon
        self.gamma = np.ones(shape)
    
    def normalize(self, x):
        """
        Apply RMSNorm to the input tensor x.
        
        Args:
            x: Input tensor (batch_size, num_features).
        
        Returns:
            Normalized tensor with the same shape as x.
        """
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.epsilon)
        return self.gamma * (x / rms)
    
    def __call__(self, x):
        return self.normalize(x)
