"""
Convolutional and pooling layers for CNN, compatible with task 4.5 architecture.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal


class ConvolutionLayer:
    """
    Convolutional layer using NumPy's convolution (scipy.signal.convolve2d).
    Compatible with layers from task 4.5.
    """

    def __init__(
        self,
        input_channels: int,
        num_filters: int,
        filter_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Initialize convolutional layer.

        Args:
            input_channels: Number of input channels (1 for greyscale)
            num_filters: Number of filters (output channels)
            filter_size: Tuple of (filter_height, filter_width)
            stride: Stride of convolution
            padding: Padding size (number of zero rows/columns)
        """
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        # Initialize filters: (num_filters, input_channels, filter_h, filter_w)
        # Use He initialization for better training
        filter_h, filter_w = filter_size
        fan_in = input_channels * filter_h * filter_w
        std = np.sqrt(2.0 / fan_in)
        self.filters = np.random.normal(
            0, std, size=(num_filters, input_channels, filter_h, filter_w)
        )

        # Initialize biases: (num_filters,)
        self.biases = np.zeros(num_filters)

        # Cache for backpropagation
        self.last_input = None
        self.last_output_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through convolution layer.

        Args:
            x: Input tensor (batch_size, input_channels, height, width)
               or (batch_size, height, width) for single channel

        Returns:
            Output tensor (batch_size, num_filters, out_height, out_width)
        """
        # Handle single channel input (batch_size, height, width)
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]  # Add channel dimension

        batch_size, input_channels, in_h, in_w = x.shape

        if input_channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {input_channels}"
            )

        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(
                x,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            x_padded = x

        # Calculate output dimensions
        filter_h, filter_w = self.filter_size
        padded_h = in_h + 2 * self.padding
        padded_w = in_w + 2 * self.padding
        out_h = (padded_h - filter_h) // self.stride + 1
        out_w = (padded_w - filter_w) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, self.num_filters, out_h, out_w))

        # Convolve each filter with each input channel
        for b in range(batch_size):
            for f in range(self.num_filters):
                filter_output = np.zeros((out_h, out_w))
                for c in range(input_channels):
                    # Use scipy.signal.convolve2d with mode='valid'
                    # Note: convolve2d performs cross-correlation by default (which is what we want for CNNs)
                    conv_result = signal.convolve2d(
                        x_padded[b, c], self.filters[f, c], mode="valid"
                    )

                    # Apply stride by subsampling
                    if self.stride > 1:
                        conv_result = conv_result[:: self.stride, :: self.stride]

                    # Ensure output size matches (handle edge cases)
                    h_slice = slice(0, min(out_h, conv_result.shape[0]))
                    w_slice = slice(0, min(out_w, conv_result.shape[1]))
                    filter_output[h_slice, w_slice] += conv_result[h_slice, w_slice]

                # Add bias
                output[b, f] = filter_output + self.biases[f]

        # Cache for backpropagation
        self.last_input = x.copy()
        self.last_output_shape = output.shape

        return output

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Calculate output shape without performing forward pass.

        Args:
            input_shape: Input shape (batch_size, input_channels, height, width)
                        or (batch_size, height, width)

        Returns:
            Output shape (batch_size, num_filters, out_height, out_width)
        """
        if len(input_shape) == 3:
            batch_size, in_h, in_w = input_shape
            input_channels = 1
        else:
            batch_size, input_channels, in_h, in_w = input_shape

        filter_h, filter_w = self.filter_size
        padded_h = in_h + 2 * self.padding
        padded_w = in_w + 2 * self.padding
        out_h = (padded_h - filter_h) // self.stride + 1
        out_w = (padded_w - filter_w) // self.stride + 1

        return (batch_size, self.num_filters, out_h, out_w)


class MaxPoolingLayer:
    """
    Max-pooling layer with configurable pooling size and stride.
    """

    def __init__(self, pool_size: Tuple[int, int] = (2, 2), stride: int = 2):
        """
        Initialize max-pooling layer.

        Args:
            pool_size: Tuple of (pool_height, pool_width)
            stride: Stride of pooling operation
        """
        self.pool_size = pool_size
        self.stride = stride

        # Cache for backpropagation
        self.last_input = None
        self.last_output = None
        self.max_indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through max-pooling layer.

        Args:
            x: Input tensor (batch_size, channels, height, width)
               or (batch_size, height, width) for single channel

        Returns:
            Output tensor with reduced spatial dimensions
        """
        # Handle single channel input
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]

        batch_size, channels, in_h, in_w = x.shape
        pool_h, pool_w = self.pool_size

        # Calculate output dimensions
        out_h = (in_h - pool_h) // self.stride + 1
        out_w = (in_w - pool_w) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, channels, out_h, out_w))
        max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=np.int32)

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        row_start = i * self.stride
                        row_end = row_start + pool_h
                        col_start = j * self.stride
                        col_end = col_start + pool_w

                        # Extract pooling region
                        pool_region = x[b, c, row_start:row_end, col_start:col_end]

                        # Find maximum value and its position
                        max_val = np.max(pool_region)
                        max_pos = np.unravel_index(
                            np.argmax(pool_region), pool_region.shape
                        )

                        output[b, c, i, j] = max_val
                        max_indices[b, c, i, j] = [
                            row_start + max_pos[0],
                            col_start + max_pos[1],
                        ]

        # Cache for backpropagation
        self.last_input = x.copy()
        self.last_output = output.copy()
        self.max_indices = max_indices

        return output

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Calculate output shape without performing forward pass.

        Args:
            input_shape: Input shape (batch_size, channels, height, width)
                        or (batch_size, height, width)

        Returns:
            Output shape with reduced spatial dimensions
        """
        if len(input_shape) == 3:
            batch_size, in_h, in_w = input_shape
            channels = 1
        else:
            batch_size, channels, in_h, in_w = input_shape

        pool_h, pool_w = self.pool_size
        out_h = (in_h - pool_h) // self.stride + 1
        out_w = (in_w - pool_w) // self.stride + 1

        return (batch_size, channels, out_h, out_w)
