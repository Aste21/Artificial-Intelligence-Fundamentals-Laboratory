"""
Convolutional Neural Network implementation.
Compatible with task 4.5 architecture patterns.
"""

import numpy as np
from typing import List, Tuple, Optional
from cnn_layers import ConvolutionLayer, MaxPoolingLayer


class ActivationFunction:
    """Activation function utilities."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU derivative: 1 if x > 0, else 0"""
        return np.where(x > 0, 1.0, 0.0)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation for output layer."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class ConvolutionalNeuralNetwork:
    """
    Convolutional Neural Network with configurable architecture.
    Supports:
    - Convolutional layers
    - ReLU activation
    - Max-pooling layers
    - Fully connected layers
    - Softmax output
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],  # (height, width) for greyscale
        conv_layers: List[dict],  # List of conv layer configs
        pool_layers: Optional[List[dict]] = None,  # List of pool layer configs
        fc_layers: List[int] = None,  # List of fully connected layer sizes
        learning_rate: float = 0.01,
    ):
        """
        Initialize CNN.

        Args:
            input_shape: Input image shape (height, width)
            conv_layers: List of dicts with keys: num_filters, filter_size, stride, padding
            pool_layers: List of dicts with keys: pool_size, stride (optional)
            fc_layers: List of fully connected layer sizes (last one is output)
            learning_rate: Learning rate for training
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate

        # Build layers
        self.conv_layers = []
        self.pool_layers = []
        self.fc_weights = []
        self.fc_biases = []

        # Track layer types and positions
        self.layer_types = []  # 'conv', 'pool', 'fc'
        self.layer_indices = {"conv": [], "pool": [], "fc": []}

        # Process convolutional layers
        current_shape = (1, input_shape[0], input_shape[1])  # (channels, h, w)
        for i, conv_config in enumerate(conv_layers):
            conv_layer = ConvolutionLayer(
                input_channels=current_shape[0],
                num_filters=conv_config["num_filters"],
                filter_size=conv_config["filter_size"],
                stride=conv_config.get("stride", 1),
                padding=conv_config.get("padding", 0),
            )
            self.conv_layers.append(conv_layer)
            self.layer_types.append("conv")
            self.layer_indices["conv"].append(len(self.layer_types) - 1)

            # Update shape
            current_shape = conv_layer.get_output_shape((1, *current_shape))[1:]

        # Process pooling layers (if any)
        if pool_layers:
            for pool_config in pool_layers:
                pool_layer = MaxPoolingLayer(
                    pool_size=pool_config.get("pool_size", (2, 2)),
                    stride=pool_config.get("stride", 2),
                )
                self.pool_layers.append(pool_layer)
                self.layer_types.append("pool")
                self.layer_indices["pool"].append(len(self.layer_types) - 1)

                # Update shape
                current_shape = pool_layer.get_output_shape((1, *current_shape))[1:]

        # Flatten for fully connected layers
        self.flatten_size = np.prod(current_shape)

        # Process fully connected layers
        if fc_layers is None:
            fc_layers = [10]  # Default: 10 output neurons

        prev_size = self.flatten_size
        for fc_size in fc_layers:
            # Initialize weights (He initialization for ReLU)
            std = np.sqrt(2.0 / prev_size)
            weights = np.random.normal(0, std, size=(fc_size, prev_size + 1))
            self.fc_weights.append(weights)
            self.fc_biases.append(np.zeros(fc_size))
            self.layer_types.append("fc")
            self.layer_indices["fc"].append(len(self.layer_types) - 1)
            prev_size = fc_size

        # Cache for backpropagation
        self.last_activations = []
        self.last_input = None

    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Forward pass through the network.

        Args:
            x: Input images (batch_size, height, width) or (batch_size, channels, height, width)

        Returns:
            Tuple of (all_activations, output)
            - all_activations: List of activations at each layer
            - output: Final output (before softmax)
        """
        if x.ndim == 2:
            # Single image: (height, width) -> (1, height, width)
            x = x[np.newaxis, :, :]
        elif x.ndim == 3 and x.shape[0] != self.input_shape[0]:
            # Assume (batch_size, height, width)
            pass
        elif x.ndim == 4:
            # Already in (batch_size, channels, height, width)
            pass

        batch_size = x.shape[0]
        self.last_input = x.copy()
        self.last_activations = []

        # Forward through conv and pool layers
        current = x
        conv_idx = 0
        pool_idx = 0

        for layer_type in self.layer_types:
            if layer_type == "conv":
                current = self.conv_layers[conv_idx].forward(current)
                # Apply ReLU activation after convolution
                current = ActivationFunction.relu(current)
                self.last_activations.append(current.copy())
                conv_idx += 1
            elif layer_type == "pool":
                current = self.pool_layers[pool_idx].forward(current)
                self.last_activations.append(current.copy())
                pool_idx += 1
            elif layer_type == "fc":
                break

        # Flatten
        batch_size = current.shape[0]
        flattened = current.reshape(batch_size, -1)

        # Forward through fully connected layers
        current_fc = flattened
        for i, (weights, bias) in enumerate(zip(self.fc_weights, self.fc_biases)):
            # Add bias column
            current_fc_with_bias = np.column_stack([np.ones(batch_size), current_fc])

            # Compute weighted sum
            output = np.dot(current_fc_with_bias, weights.T)

            # Apply activation (ReLU for hidden, none for output - softmax applied in predict)
            if i < len(self.fc_weights) - 1:
                # Hidden layer: ReLU
                output = ActivationFunction.relu(output)

            current_fc = output
            self.last_activations.append(output.copy())

        return self.last_activations, current_fc

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            x: Input images (batch_size, height, width)

        Returns:
            Predicted class labels (batch_size,)
        """
        _, output = self.forward(x)
        # Apply softmax
        proba = ActivationFunction.softmax(output)
        return np.argmax(proba, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            x: Input images (batch_size, height, width)

        Returns:
            Class probabilities (batch_size, n_classes)
        """
        _, output = self.forward(x)
        return ActivationFunction.softmax(output)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the CNN using backpropagation.

        Args:
            x: Input images (n_samples, height, width)
            y: Class labels (n_samples,) - values 0 to n_classes-1
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress

        Returns:
            List of losses per epoch
        """
        n_samples = x.shape[0]
        n_classes = self.fc_weights[-1].shape[0]

        # Convert labels to one-hot
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1.0

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y_onehot[indices]

            # Train in batches
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                x_batch = x_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                # Forward pass
                activations, output = self.forward(x_batch)

                # Compute loss (cross-entropy)
                proba = ActivationFunction.softmax(output)
                loss = -np.mean(np.sum(y_batch * np.log(proba + 1e-10), axis=1))
                epoch_loss += loss

                # Backpropagation
                # Output layer gradient
                delta = proba - y_batch  # (batch_size, n_classes)

                # Get FC layer activations
                num_conv_pool = len(self.conv_layers) + len(self.pool_layers)
                fc_activations = activations[num_conv_pool:]

                # Get FC inputs (with bias)
                conv_pool_output = (
                    activations[num_conv_pool - 1] if num_conv_pool > 0 else x_batch
                )
                batch_size_batch = conv_pool_output.shape[0]

                # Flatten conv/pool output
                if len(conv_pool_output.shape) == 4:
                    flattened = conv_pool_output.reshape(batch_size_batch, -1)
                elif len(conv_pool_output.shape) == 3:
                    flattened = conv_pool_output.reshape(batch_size_batch, -1)
                else:
                    flattened = conv_pool_output

                # Build FC inputs list
                fc_inputs = []
                fc_inputs.append(
                    np.column_stack([np.ones(batch_size_batch), flattened])
                )

                for i in range(len(self.fc_weights) - 1):
                    fc_inputs.append(
                        np.column_stack([np.ones(batch_size_batch), fc_activations[i]])
                    )

                # Backprop through FC layers
                current_delta = delta
                for i in range(len(self.fc_weights) - 1, -1, -1):
                    if i < len(self.fc_weights) - 1:
                        # Hidden layer: propagate error and apply ReLU derivative
                        next_weights = self.fc_weights[i + 1][:, 1:]  # Remove bias
                        propagated = np.dot(current_delta, next_weights)
                        relu_deriv = ActivationFunction.relu_derivative(
                            fc_activations[i]
                        )
                        current_delta = propagated * relu_deriv

                    # Update weights
                    weight_update = (
                        self.learning_rate
                        * np.dot(current_delta.T, fc_inputs[i])
                        / batch_size_batch
                    )
                    self.fc_weights[i] -= weight_update
                    bias_update = self.learning_rate * np.mean(current_delta, axis=0)
                    self.fc_biases[i] -= bias_update

                # Note: Full backpropagation through conv layers would require
                # implementing backward passes for conv and pool layers.
                # For this implementation, we focus on FC layer training.
                # In practice, you'd want to implement full backprop through conv layers.

            avg_loss = epoch_loss / (
                n_samples // batch_size + (1 if n_samples % batch_size else 0)
            )
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses
