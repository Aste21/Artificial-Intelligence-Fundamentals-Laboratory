"""
Implementation of a shallow fully connected neural network (up to 5 layers).
Supports configurable architecture, logistic and ReLU activation functions,
and training using backpropagation.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from enum import Enum


class ActivationFunction(Enum):
    """Enumeration of available activation functions."""

    LOGISTIC = "logistic"  # Sigmoid
    RELU = "relu"


class NeuralNetwork:
    """
    Shallow fully connected neural network (up to 5 layers).

    The network uses backpropagation for training:
    Δw_k^j = η * δ_k * f'(s_k) * x_l = η * Σ_i(d_i - y_i) * f'(s_i) * W_i^k * f'(s_k) * x_l

    where:
    - w_k^j are weights from layer j to neuron k
    - η is the learning rate
    - δ_k is the error signal for neuron k
    - f'(s) is the derivative of the activation function
    - d_i is the expected output
    - y_i is the predicted output

    Note on ReLU as output activation:
    ReLU outputs are unbounded [0, +∞), which can cause issues with binary classification.
    For better results, use logistic activation for the output layer. If using ReLU,
    the network will normalize outputs using softmax for predictions.
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int],
        activations: List[ActivationFunction],
        learning_rate: float = 0.1,
        beta: float = 1.0,
    ):
        """
        Initialize the neural network.

        Args:
            input_dim: Number of input features
            layer_sizes: List of neuron counts for each layer (excluding input)
                        Must have at least 2 layers (hidden + output)
                        Total layers (including input) must be <= 5
            activations: List of activation functions for each layer
                        Length should be len(layer_sizes)
            learning_rate: Learning rate (η)
            beta: Beta parameter for logistic activation
        """
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least 2 layers (hidden + output)")

        if len(layer_sizes) + 1 > 5:  # +1 for input layer
            raise ValueError(
                "Network can have at most 5 layers total (including input)"
            )

        if len(activations) != len(layer_sizes):
            raise ValueError(
                "Number of activation functions must match number of layers"
            )

        # Validate that input layer has multiple neurons (>2) for 2D input
        if input_dim == 2:
            if layer_sizes[0] <= 2:
                raise ValueError(
                    "For 2D input, first hidden layer must have more than 2 neurons"
                )

        # Output layer size is configurable (2 for binary classification, 10 for MNIST, etc.)
        # No validation needed - user specifies the number of output neurons

        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.learning_rate = learning_rate
        self.beta = beta

        # Build network architecture
        # Weights: list of weight matrices, one per layer
        # Each weight matrix W[i] has shape (layer_sizes[i], layer_sizes[i-1] + 1)
        # The +1 is for bias term
        self.weights = []
        self.biases = []

        # Initialize weights - use smaller range for ReLU to avoid dying neurons
        # For ReLU, use He initialization (smaller weights) to prevent dying neurons
        for i in range(len(layer_sizes)):
            if i == 0:
                prev_size = input_dim
            else:
                prev_size = layer_sizes[i - 1]

            # Use smaller initialization for ReLU layers
            if i < len(activations) and activations[i] == ActivationFunction.RELU:
                # He initialization for ReLU: sqrt(2 / n)
                std = np.sqrt(2.0 / (prev_size + 1))
                self.weights.append(
                    np.random.normal(0, std, size=(layer_sizes[i], prev_size + 1))
                )
            else:
                # Standard initialization for logistic
                self.weights.append(
                    np.random.uniform(-0.5, 0.5, size=(layer_sizes[i], prev_size + 1))
                )

        # Get activation functions and derivatives
        self.activation_funcs = [
            self._get_activation_function(act) for act in activations
        ]
        self.activation_derivatives = [
            self._get_activation_derivative(act) for act in activations
        ]

    def reset_weights(self):
        """Reset all weights to random initial values."""
        self.weights = []
        for i in range(len(self.layer_sizes)):
            if i == 0:
                prev_size = self.input_dim
            else:
                prev_size = self.layer_sizes[i - 1]

            # Use smaller initialization for ReLU layers
            if (
                i < len(self.activations)
                and self.activations[i] == ActivationFunction.RELU
            ):
                # He initialization for ReLU
                std = np.sqrt(2.0 / (prev_size + 1))
                self.weights.append(
                    np.random.normal(0, std, size=(self.layer_sizes[i], prev_size + 1))
                )
            else:
                # Standard initialization for logistic
                self.weights.append(
                    np.random.uniform(
                        -0.5, 0.5, size=(self.layer_sizes[i], prev_size + 1)
                    )
                )

    def _get_activation_function(self, activation: ActivationFunction) -> Callable:
        """Get the activation function."""
        if activation == ActivationFunction.LOGISTIC:
            return lambda s: 1.0 / (1.0 + np.exp(-self.beta * s))
        elif activation == ActivationFunction.RELU:
            return lambda s: np.maximum(0, s)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def _get_activation_derivative(self, activation: ActivationFunction) -> Callable:
        """Get the derivative of the activation function."""
        if activation == ActivationFunction.LOGISTIC:
            # Derivative of logistic: β * f(s) * (1 - f(s))
            return (
                lambda s: self.beta
                * (1.0 / (1.0 + np.exp(-self.beta * s)))
                * (1.0 - (1.0 / (1.0 + np.exp(-self.beta * s))))
            )
        elif activation == ActivationFunction.RELU:
            return lambda s: np.where(s > 0, 1.0, 0.0)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network.

        Args:
            x: Input samples (n_samples, input_dim)

        Returns:
            Tuple of (activations, weighted_sums)
            - activations: List of activation outputs for each layer (with bias for next layer)
            - weighted_sums: List of weighted sums (s) for each layer
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        activations = []
        weighted_sums = []

        # Forward through each layer
        current_input = x
        for i in range(len(self.layer_sizes)):
            # Add bias to current input
            current_input_with_bias = np.column_stack(
                [np.ones(current_input.shape[0]), current_input]
            )

            # Store input with bias (for backpropagation)
            if i == 0:
                # For first layer, also store input
                activations.append(current_input_with_bias)

            # Compute weighted sum: s = x_with_bias @ W^T
            # weights[i] shape: (layer_sizes[i], prev_layer_size + 1)
            # current_input_with_bias shape: (batch_size, prev_layer_size + 1)
            # s shape: (batch_size, layer_sizes[i])
            s = np.dot(current_input_with_bias, self.weights[i].T)
            weighted_sums.append(s)

            # Apply activation function
            current_input = self.activation_funcs[i](s)

            # Store activation (with bias for next layer)
            current_input_with_bias = np.column_stack(
                [np.ones(current_input.shape[0]), current_input]
            )
            activations.append(current_input_with_bias)

        return activations, weighted_sums

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the network output (continuous values).

        Args:
            x: Input samples (n_samples, input_dim)

        Returns:
            Output values (n_samples, output_dim)
        """
        activations, _ = self.forward(x)
        # Return output layer activations (without bias)
        return activations[-1][:, 1:]  # Remove bias column

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            x: Input samples (n_samples, input_dim)

        Returns:
            Predicted class labels (n_samples,) - class with highest confidence
        """
        outputs = self.evaluate(x)

        # For ReLU output, we need to handle it differently
        # ReLU outputs are [0, +∞), so we need to normalize or use difference
        if self.activations[-1] == ActivationFunction.RELU:
            # For ReLU, use the difference between outputs as decision criterion
            # If output[0] > output[1], predict class 0, else class 1
            # But better: normalize outputs first to make them comparable
            # Use softmax-like normalization for ReLU outputs
            exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            normalized = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
            return np.argmax(normalized, axis=1)
        else:
            # For logistic, outputs are already in [0,1] range
            return np.argmax(outputs, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (confidence values).

        Args:
            x: Input samples (n_samples, input_dim)

        Returns:
            Confidence values for each class (n_samples, n_classes)
        """
        outputs = self.evaluate(x)

        # Use softmax to get proper probabilities that sum to 1
        # This works for both ReLU and logistic outputs
        # For ReLU: outputs are unbounded [0, +∞), softmax normalizes them
        # For logistic: outputs are in [0,1] but don't sum to 1, softmax fixes that
        # Subtract max for numerical stability to prevent overflow
        shifted_outputs = outputs - np.max(outputs, axis=1, keepdims=True)
        exp_outputs = np.exp(shifted_outputs)
        proba = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)

        return proba

    def train(
        self,
        x: np.ndarray,
        d: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        variable_lr: bool = False,
        eta_min: float = 0.01,
        eta_max: float = 0.1,
    ) -> List[float]:
        """
        Train the network using backpropagation.

        Args:
            x: Input samples (n_samples, input_dim)
            d: Expected class labels (n_samples,) - should be 0 or 1
            epochs: Number of training epochs
            batch_size: Batch size for training (None = full batch)
            variable_lr: Whether to use variable learning rate
            eta_min: Minimum learning rate (for variable LR)
            eta_max: Maximum learning rate (for variable LR)

        Returns:
            List of errors per epoch
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Convert labels to one-hot encoding
        # Determine number of classes from output layer size
        n_classes = self.layer_sizes[-1]
        d_onehot = np.zeros((len(d), n_classes))
        d_onehot[np.arange(len(d)), d.astype(int)] = 1.0

        errors_per_epoch = []
        n_samples = x.shape[0]

        # Determine batch size
        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
            # Calculate learning rate for this epoch
            if variable_lr:
                eta = eta_min + (eta_max - eta_min) * (
                    1 + np.cos(epoch / epochs * np.pi)
                )
            else:
                eta = self.learning_rate

            total_error = 0.0

            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            d_shuffled = d_onehot[indices]

            # Train in batches
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                x_batch = x_shuffled[batch_start:batch_end]
                d_batch = d_shuffled[batch_start:batch_end]

                # Forward pass
                activations, weighted_sums = self.forward(x_batch)

                # Get output (without bias)
                y = activations[-1][:, 1:]

                # For ReLU output, we need to handle unbounded outputs
                # Normalize outputs for error computation to make training more stable
                if self.activations[-1] == ActivationFunction.RELU:
                    # Clip ReLU outputs to reasonable range for error computation
                    # This helps with training stability
                    y_clipped = np.clip(y, 0, 10)  # Clip to [0, 10] for stability
                    error = d_batch - y_clipped
                else:
                    # For logistic, outputs are already in [0,1]
                    error = d_batch - y

                total_error += np.sum(error**2)

                # Backpropagation
                # According to formula: Δw_k^j = η * δ_k * f'(s_k) * x_l
                # where δ_k = Σ_i(d_i - y_i) * f'(s_i) * W_i^k * f'(s_k) for hidden layers
                # For output layer: δ_k = (d_k - y_k) * f'(s_k)

                # Start with output layer
                # δ_k = (d_k - y_k) * f'(s_k)
                output_layer_idx = len(self.layer_sizes) - 1
                f_prime_output = self.activation_derivatives[output_layer_idx](
                    weighted_sums[output_layer_idx]
                )
                delta = error * f_prime_output  # (batch_size, output_dim)

                # Store deltas for each layer
                deltas = [None] * len(self.layer_sizes)
                deltas[output_layer_idx] = delta

                # Backpropagate through hidden layers
                for i in range(output_layer_idx - 1, -1, -1):
                    # δ_k = Σ_i(δ_i * W_i^k) * f'(s_k)
                    # Get next layer's delta and weights
                    next_delta = deltas[i + 1]  # (batch_size, next_layer_size)
                    next_weights = self.weights[i + 1][:, 1:]  # Remove bias column

                    # Propagate error: Σ_i(δ_i * W_i^k)
                    propagated_error = np.dot(
                        next_delta, next_weights
                    )  # (batch_size, current_layer_size)

                    # Multiply by derivative: * f'(s_k)
                    f_prime = self.activation_derivatives[i](weighted_sums[i])
                    delta = propagated_error * f_prime
                    deltas[i] = delta

                # Update weights for all layers
                for i in range(len(self.layer_sizes) - 1, -1, -1):
                    # Get previous layer activations (with bias)
                    prev_activations = activations[i]

                    # Update weights: ΔW = η * δ^T * x
                    # delta shape: (batch_size, layer_size)
                    # prev_activations shape: (batch_size, prev_layer_size + 1)
                    # Weight update: (layer_size, prev_layer_size + 1)
                    # Average over batch
                    batch_len = len(x_batch)
                    weight_update = (
                        eta * np.dot(deltas[i].T, prev_activations) / batch_len
                    )
                    self.weights[i] += weight_update

            errors_per_epoch.append(total_error / n_samples)

        return errors_per_epoch
