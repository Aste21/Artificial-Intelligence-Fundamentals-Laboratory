"""
Implementation of a single artificial neuron with various activation functions.
Supports training using the delta rule with different activation functions.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from enum import Enum


class ActivationFunction(Enum):
    """Enumeration of available activation functions."""
    HEAVISIDE = "heaviside"
    SIGMOID = "sigmoid"
    SIN = "sin"
    TANH = "tanh"
    SIGN = "sign"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"


class Neuron:
    """
    Single artificial neuron that can be trained using the delta rule.
    
    The neuron uses the training formula:
    Δw_j = η * ε * f'(s) * x_j = η * (d - y) * f'(w^T * x_j) * x_j
    
    where:
    - x_j is the j-th sample
    - f'(s) is the derivative of the activation function
    - w are the weights (including bias)
    - η is the learning rate
    - d is the expected class label
    - y is the predicted class label
    """
    
    def __init__(self, input_dim: int, activation: ActivationFunction = ActivationFunction.SIGMOID, 
                 learning_rate: float = 0.1, beta: float = 1.0):
        """
        Initialize the neuron.
        
        Args:
            input_dim: Number of input features (without bias)
            activation: Activation function to use
            learning_rate: Initial learning rate (η)
            beta: Beta parameter for sigmoid function
        """
        # Weights include bias term (w_0 for bias, w_1...w_n for inputs)
        self.weights = np.random.uniform(-0.5, 0.5, size=input_dim + 1)
        self.activation_type = activation
        self.learning_rate = learning_rate
        self.beta = beta
        self.leaky_relu_alpha = 0.01
        
        # Get activation function and its derivative
        self.activation_func = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
    
    def _get_activation_function(self, activation: ActivationFunction) -> Callable:
        """Get the activation function."""
        if activation == ActivationFunction.HEAVISIDE:
            return lambda s: np.where(s >= 0, 1.0, 0.0)
        elif activation == ActivationFunction.SIGMOID:
            return lambda s: 1.0 / (1.0 + np.exp(-self.beta * s))
        elif activation == ActivationFunction.SIN:
            return lambda s: np.sin(s)
        elif activation == ActivationFunction.TANH:
            return lambda s: np.tanh(s)
        elif activation == ActivationFunction.SIGN:
            return lambda s: np.sign(s)
        elif activation == ActivationFunction.RELU:
            return lambda s: np.maximum(0, s)
        elif activation == ActivationFunction.LEAKY_RELU:
            return lambda s: np.where(s > 0, s, self.leaky_relu_alpha * s)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def _get_activation_derivative(self, activation: ActivationFunction) -> Callable:
        """Get the derivative of the activation function."""
        if activation == ActivationFunction.HEAVISIDE:
            # For Heaviside, derivative is assumed to be 1 for training
            return lambda s: np.ones_like(s)
        elif activation == ActivationFunction.SIGMOID:
            return lambda s: self.beta * self.activation_func(s) * (1.0 - self.activation_func(s))
        elif activation == ActivationFunction.SIN:
            return lambda s: np.cos(s)
        elif activation == ActivationFunction.TANH:
            return lambda s: 1.0 - np.tanh(s) ** 2
        elif activation == ActivationFunction.SIGN:
            # Sign function derivative is 0 almost everywhere, use small value for training
            return lambda s: np.where(np.abs(s) < 1e-10, 1.0, 0.0)
        elif activation == ActivationFunction.RELU:
            return lambda s: np.where(s > 0, 1.0, 0.0)
        elif activation == ActivationFunction.LEAKY_RELU:
            return lambda s: np.where(s > 0, 1.0, self.leaky_relu_alpha)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.
        
        Args:
            x: Input samples (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Add bias term (column of ones)
        x_with_bias = np.column_stack([np.ones(x.shape[0]), x])
        
        # Compute weighted sum: s = w^T * x
        s = np.dot(x_with_bias, self.weights)
        
        # Apply activation function
        y = self.activation_func(s)
        
        # For classification, map to [0, 1] class labels
        if self.activation_type == ActivationFunction.HEAVISIDE:
            # Already in [0, 1]
            return y
        elif self.activation_type == ActivationFunction.SIGMOID:
            # Already in [0, 1], threshold at 0.5
            return (y >= 0.5).astype(float)
        elif self.activation_type == ActivationFunction.TANH:
            # Range [-1, 1], map to [0, 1] by thresholding at 0
            return (y >= 0).astype(float)
        elif self.activation_type == ActivationFunction.SIN:
            # Range [-1, 1], map to [0, 1] by thresholding at 0
            return (y >= 0).astype(float)
        elif self.activation_type == ActivationFunction.SIGN:
            # Range [-1, 0, 1], map to [0, 1] by thresholding at 0
            return (y >= 0).astype(float)
        elif self.activation_type == ActivationFunction.RELU:
            # Range [0, inf), threshold at some small value or use > 0
            return (y > 0).astype(float)
        elif self.activation_type == ActivationFunction.LEAKY_RELU:
            # Range (-inf, inf), but mostly positive, threshold at 0
            return (y >= 0).astype(float)
        else:
            # Default: threshold at 0
            return (y >= 0).astype(float)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the neuron output (continuous values, not thresholded).
        
        Args:
            x: Input samples (n_samples, n_features)
            
        Returns:
            Continuous output values
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        x_with_bias = np.column_stack([np.ones(x.shape[0]), x])
        s = np.dot(x_with_bias, self.weights)
        return self.activation_func(s)
    
    def train(self, x: np.ndarray, d: np.ndarray, epochs: int = 100, 
              variable_lr: bool = False, eta_min: float = 0.01, eta_max: float = 0.1) -> list:
        """
        Train the neuron using the delta rule.
        
        Args:
            x: Input samples (n_samples, n_features)
            d: Expected class labels (n_samples,)
            epochs: Number of training epochs
            variable_lr: Whether to use variable learning rate
            eta_min: Minimum learning rate (for variable LR)
            eta_max: Maximum learning rate (for variable LR)
            
        Returns:
            List of errors per epoch
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        errors_per_epoch = []
        n_samples = x.shape[0]
        
        for epoch in range(epochs):
            # Calculate learning rate for this epoch
            if variable_lr:
                eta = eta_min + (eta_max - eta_min) * (1 + np.cos(epoch / epochs * np.pi))
            else:
                eta = self.learning_rate
            
            total_error = 0.0
            
            # Train on all samples
            for j in range(n_samples):
                x_j = x[j]
                d_j = d[j]
                
                # Add bias term
                x_j_with_bias = np.concatenate([[1.0], x_j])
                
                # Compute weighted sum: s = w^T * x_j
                s = np.dot(self.weights, x_j_with_bias)
                
                # Compute output: y = f(s)
                y = self.activation_func(s)
                
                # Compute error: ε = d - y
                error = d_j - y
                total_error += error ** 2
                
                # Compute derivative: f'(s)
                f_prime = self.activation_derivative(s)
                
                # Update weights: Δw_j = η * ε * f'(s) * x_j
                delta_w = eta * error * f_prime * x_j_with_bias
                self.weights += delta_w
            
            errors_per_epoch.append(total_error / n_samples)
        
        return errors_per_epoch
    
    def get_decision_boundary_params(self) -> Tuple[float, float, float]:
        """
        Get parameters for decision boundary visualization.
        For 2D input: w0 + w1*x + w2*y = 0
        Returns: (a, b, c) where ax + by + c = 0
        """
        if len(self.weights) != 3:
            raise ValueError("Decision boundary visualization only works for 2D inputs")
        
        # w0 (bias) + w1*x + w2*y = 0
        # Rearranged: w2*y = -w0 - w1*x
        # y = (-w0 - w1*x) / w2
        # Or in form: w1*x + w2*y + w0 = 0
        return (self.weights[1], self.weights[2], self.weights[0])
