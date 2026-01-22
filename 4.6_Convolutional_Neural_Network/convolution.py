"""
Custom convolution operator for 2D greyscale images.
Implements convolution with stride and padding support.
"""

import numpy as np
from typing import Tuple


def convolve_2d(
    input_image: np.ndarray,
    filter_kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    Apply 2D convolution to a greyscale image.

    Args:
        input_image: 2D numpy array (height, width) - greyscale image
        filter_kernel: 2D numpy array (filter_height, filter_width) - filter to apply
        stride: Stride of the convolution (default: 1)
        padding: Number of zero-filled rows/columns to add around the input (default: 0)

    Returns:
        2D numpy array - convolved output image

    Example:
        input_image = np.array([[1, 1, 1, 0, 0],
                               [0, 1, 1, 1, 0],
                               [0, 0, 1, 1, 1],
                               [0, 0, 1, 1, 0],
                               [0, 1, 1, 0, 0]])
        filter_kernel = np.array([[1, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 1]])
        output = convolve_2d(input_image, filter_kernel, stride=1, padding=0)
        # Expected output: [[4, 3, 4],
        #                   [2, 4, 3],
        #                   [2, 3, 4]]
    """
    # Validate inputs
    if input_image.ndim != 2:
        raise ValueError(f"input_image must be 2D, got {input_image.ndim}D")
    if filter_kernel.ndim != 2:
        raise ValueError(f"filter_kernel must be 2D, got {filter_kernel.ndim}D")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")

    # Get dimensions
    img_h, img_w = input_image.shape
    filter_h, filter_w = filter_kernel.shape

    # Apply padding
    if padding > 0:
        padded_image = np.pad(
            input_image,
            pad_width=((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
    else:
        padded_image = input_image

    # Calculate output dimensions
    # output_h = floor((img_h + 2*padding - filter_h) / stride) + 1
    # output_w = floor((img_w + 2*padding - filter_w) / stride) + 1
    padded_h, padded_w = padded_image.shape
    output_h = (padded_h - filter_h) // stride + 1
    output_w = (padded_w - filter_w) // stride + 1

    # Initialize output
    output = np.zeros((output_h, output_w))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # Calculate the top-left corner of the filter position
            row_start = i * stride
            row_end = row_start + filter_h
            col_start = j * stride
            col_end = col_start + filter_w

            # Extract the image patch
            image_patch = padded_image[row_start:row_end, col_start:col_end]

            # Compute element-wise product and sum
            output[i, j] = np.sum(image_patch * filter_kernel)

    return output


def test_convolution_example():
    """Test the convolution with the provided example."""
    input_image = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ]
    )

    filter_kernel = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

    output = convolve_2d(input_image, filter_kernel, stride=1, padding=0)

    expected_output = np.array([[4, 3, 4], [2, 4, 3], [2, 3, 4]])

    print("Input image:")
    print(input_image)
    print("\nFilter:")
    print(filter_kernel)
    print("\nOutput:")
    print(output)
    print("\nExpected output:")
    print(expected_output)
    print(f"\nMatch: {np.allclose(output, expected_output)}")

    return np.allclose(output, expected_output)


if __name__ == "__main__":
    test_convolution_example()
