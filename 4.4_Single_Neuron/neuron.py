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
    
    def __init__(self, input_dim: int, 
                 training_activation: ActivationFunction = ActivationFunction.SIGMOID,
                 evaluation_activation: Optional[ActivationFunction] = None,
                 learning_rate: float = 0.1, beta: float = 1.0):
        """
        Initialize the neuron.
        
        Args:
            input_dim: Number of input features (without bias)
            training_activation: Activation function to use for training (Heaviside, Sigmoid, Sin, Tanh)
            evaluation_activation: Activation function to use for evaluation/prediction (all functions)
                                   If None, uses training_activation
            learning_rate: Initial learning rate (η)
            beta: Beta parameter for sigmoid function
        """
        # Weights include bias term (w_0 for bias, w_1...w_n for inputs)
        self.weights = np.random.uniform(-0.5, 0.5, size=input_dim + 1)
        self.training_activation = training_activation
        self.evaluation_activation = evaluation_activation if evaluation_activation is not None else training_activation
        self.learning_rate = learning_rate
        self.beta = beta
        self.leaky_relu_alpha = 0.01
        self.input_dim = input_dim
        
        # Validate training activation
        valid_training_activations = [
            ActivationFunction.HEAVISIDE,
            ActivationFunction.SIGMOID,
            ActivationFunction.SIN,
            ActivationFunction.TANH
        ]
        if training_activation not in valid_training_activations:
            raise ValueError(
                f"Training activation must be one of: {[a.value for a in valid_training_activations]}. "
                f"Got: {training_activation.value}"
            )
        
        # Get activation functions and derivative for training
        self.training_activation_func = self._get_activation_function(training_activation)
        self.training_activation_derivative = self._get_activation_derivative(training_activation)
        
        # Get activation function for evaluation
        self.evaluation_activation_func = self._get_activation_function(self.evaluation_activation)
    
    def reset_weights(self):
        """Reset weights to random initial values."""
        self.weights = np.random.uniform(-0.5, 0.5, size=self.input_dim + 1)
    
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
            # Derivative of sigmoid: β * f(s) * (1 - f(s))
            # where f(s) = 1 / (1 + exp(-β*s))
            return lambda s: self.beta * (1.0 / (1.0 + np.exp(-self.beta * s))) * (1.0 - (1.0 / (1.0 + np.exp(-self.beta * s))))
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
        Predict class labels for input samples using evaluation activation function.
        
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
        
        # Apply evaluation activation function
        y = self.evaluation_activation_func(s)
        
        # For classification, map to [0, 1] class labels
        if self.evaluation_activation == ActivationFunction.HEAVISIDE:
            # Already in [0, 1]
            return y
        elif self.evaluation_activation == ActivationFunction.SIGMOID:
            # Already in [0, 1], threshold at 0.5
            return (y >= 0.5).astype(float)
        elif self.evaluation_activation == ActivationFunction.TANH:
            # Range [-1, 1], map to [0, 1] by thresholding at 0
            return (y >= 0).astype(float)
        elif self.evaluation_activation == ActivationFunction.SIN:
            # Range [-1, 1], map to [0, 1] by thresholding at 0
            return (y >= 0).astype(float)
        elif self.evaluation_activation == ActivationFunction.SIGN:
            # Range [-1, 0, 1], map to [0, 1] by thresholding at 0
            return (y >= 0).astype(float)
        elif self.evaluation_activation == ActivationFunction.RELU:
            # Range [0, inf), threshold at some small value or use > 0
            return (y > 0).astype(float)
        elif self.evaluation_activation == ActivationFunction.LEAKY_RELU:
            # Range (-inf, inf), but mostly positive, threshold at 0
            return (y >= 0).astype(float)
        else:
            # Default: threshold at 0
            return (y >= 0).astype(float)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the neuron output (continuous values, not thresholded) using evaluation activation function.
        
        Args:
            x: Input samples (n_samples, n_features)
            
        Returns:
            Continuous output values
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        x_with_bias = np.column_stack([np.ones(x.shape[0]), x])
        s = np.dot(x_with_bias, self.weights)
        return self.evaluation_activation_func(s)
    
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
                
                # Compute output: y = f(s) using training activation function
                y = self.training_activation_func(s)
                
                # Normalize output to [0, 1] range for SIN and TANH
                # These functions output [-1, 1], but expected values d are [0, 1]
                if self.training_activation == ActivationFunction.SIN:
                    # Normalize sin output: [-1, 1] -> [0, 1]
                    y = (y + 1.0) / 2.0
                elif self.training_activation == ActivationFunction.TANH:
                    # Normalize tanh output: [-1, 1] -> [0, 1]
                    y = (y + 1.0) / 2.0
                
                # Compute error: ε = d - y
                error = d_j - y
                total_error += error ** 2
                
                # Compute derivative: f'(s) using training activation derivative
                f_prime = self.training_activation_derivative(s)
                
                # For SIN and TANH, we normalized the output, so we need to normalize the derivative too
                # If y_normalized = (y + 1) / 2, then d/ds y_normalized = f'(s) / 2
                if self.training_activation == ActivationFunction.SIN:
                    f_prime = f_prime / 2.0
                elif self.training_activation == ActivationFunction.TANH:
                    f_prime = f_prime / 2.0
                
                # Update weights: Δw_j = η * ε * f'(s) * x_j
                delta_w = eta * error * f_prime * x_j_with_bias
                self.weights += delta_w
            
            errors_per_epoch.append(total_error / n_samples)
        
        return errors_per_epoch
    
    def get_decision_boundary_threshold(self) -> float:
        """
        Get the threshold value in the activation function output space
        that corresponds to the decision boundary.
        Uses evaluation activation function for visualization.
        
        Returns:
            Threshold value in the output space of the activation function
        """
        if self.evaluation_activation == ActivationFunction.HEAVISIDE:
            return 0.5
        elif self.evaluation_activation == ActivationFunction.SIGMOID:
            return 0.5
        elif self.evaluation_activation == ActivationFunction.TANH:
            return 0.0
        elif self.evaluation_activation == ActivationFunction.SIN:
            return 0.0
        elif self.evaluation_activation == ActivationFunction.SIGN:
            return 0.0
        elif self.evaluation_activation == ActivationFunction.RELU:
            return 0.0
        elif self.evaluation_activation == ActivationFunction.LEAKY_RELU:
            return 0.0
        else:
            return 0.5
    
    def get_decision_boundary_params(self) -> Tuple[float, float, float]:
        """
        Get parameters for decision boundary visualization.
        For a single neuron, the decision boundary is always linear.
        
        The decision boundary is where: f(w0 + w1*x + w2*y) = threshold
        For monotonic activation functions, this becomes: w0 + w1*x + w2*y = f^(-1)(threshold)
        
        Returns: (a, b, c) where ax + by + c = 0 represents the decision boundary
        """
        if len(self.weights) != 3:
            raise ValueError("Decision boundary visualization only works for 2D inputs")
        
        threshold_output = self.get_decision_boundary_threshold()
        
        # For most activation functions, we need to find the input value s
        # such that f(s) = threshold_output
        # Then the boundary is: w0 + w1*x + w2*y = s
        
        # Calculate s (the input to activation function at the boundary)
        # Use evaluation activation for visualization
        if self.evaluation_activation == ActivationFunction.HEAVISIDE:
            # f(s) = 0.5 when s = 0
            s_threshold = 0.0
        elif self.evaluation_activation == ActivationFunction.SIGMOID:
            # f(s) = 0.5 when s = 0 (for beta=1, or s = -ln(1/threshold - 1) / beta)
            # For threshold = 0.5: s = 0
            s_threshold = 0.0
        elif self.evaluation_activation == ActivationFunction.TANH:
            # f(s) = 0 when s = 0
            s_threshold = 0.0
        elif self.evaluation_activation == ActivationFunction.SIN:
            # f(s) = 0 when s = 0 (or s = k*pi, but we use 0)
            s_threshold = 0.0
        elif self.evaluation_activation == ActivationFunction.SIGN:
            # f(s) = 0 when s = 0
            s_threshold = 0.0
        elif self.evaluation_activation == ActivationFunction.RELU:
            # f(s) = 0 when s = 0
            s_threshold = 0.0
        elif self.evaluation_activation == ActivationFunction.LEAKY_RELU:
            # f(s) = 0 when s = 0
            s_threshold = 0.0
        else:
            s_threshold = 0.0
        
        # The decision boundary equation: w0 + w1*x + w2*y = s_threshold
        # Rearranged: w1*x + w2*y + (w0 - s_threshold) = 0
        # So: a = w1, b = w2, c = w0 - s_threshold
        return (self.weights[1], self.weights[2], self.weights[0] - s_threshold)
