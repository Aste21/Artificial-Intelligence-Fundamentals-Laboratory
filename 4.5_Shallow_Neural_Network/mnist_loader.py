"""
MNIST dataset loader for all 10 digits (0-9).
"""

import numpy as np
import os
from typing import Tuple, Optional
import kagglehub


def load_mnist(path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST dataset (all 10 digits: 0-9).

    Args:
        path: Path to dataset. If None, downloads using kagglehub.

    Returns:
        Tuple of (images, labels) where:
        - images: Flattened pixel vectors (n_samples, 784)
        - labels: Class labels (n_samples,) - values 0-9 for digits 0-9
    """
    if path is None:
        # Download dataset using kagglehub
        try:
            path = kagglehub.dataset_download("hojjatk/mnist-dataset")
            print(f"Dataset downloaded to: {path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MNIST dataset: {e}. Make sure kagglehub is installed and you have internet connection."
            )

    # Try to find the data files in the downloaded directory
    # The structure might vary, so we'll search for the files
    train_images_path = None
    train_labels_path = None
    test_images_path = None
    test_labels_path = None

    # Search for files
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "train-images.idx3-ubyte" or "train-images" in file.lower():
                train_images_path = os.path.join(root, file)
            elif file == "train-labels.idx1-ubyte" or "train-labels" in file.lower():
                train_labels_path = os.path.join(root, file)
            elif file == "t10k-images.idx3-ubyte" or "test-images" in file.lower():
                test_images_path = os.path.join(root, file)
            elif file == "t10k-labels.idx1-ubyte" or "test-labels" in file.lower():
                test_labels_path = os.path.join(root, file)

    # If not found, try direct paths
    if train_images_path is None:
        train_images_path = os.path.join(path, "train-images.idx3-ubyte")
    if train_labels_path is None:
        train_labels_path = os.path.join(path, "train-labels.idx1-ubyte")
    if test_images_path is None:
        test_images_path = os.path.join(path, "t10k-images.idx3-ubyte")
    if test_labels_path is None:
        test_labels_path = os.path.join(path, "t10k-labels.idx1-ubyte")

    # Load images and labels
    try:
        train_images = _load_images(train_images_path)
        train_labels = _load_labels(train_labels_path)
        test_images = _load_images(test_images_path)
        test_labels = _load_labels(test_labels_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"MNIST data files not found. Expected files:\n"
            f"  - {train_images_path}\n"
            f"  - {train_labels_path}\n"
            f"  - {test_images_path}\n"
            f"  - {test_labels_path}\n"
            f"Please check the dataset path or re-download the dataset."
        ) from e

    # Use all digits (0-9), no filtering needed
    # Labels are already 0-9, no conversion needed

    # Combine train and test
    all_images = np.vstack([train_images, test_images])
    all_labels = np.hstack([train_labels, test_labels])

    # Normalize pixel values to [0, 1]
    all_images = all_images.astype(np.float32) / 255.0

    return all_images, all_labels


def _load_images(filepath: str) -> np.ndarray:
    """Load MNIST images from IDX file format."""
    with open(filepath, "rb") as f:
        # Read magic number
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")

        # Read number of images
        n_images = int.from_bytes(f.read(4), "big")

        # Read dimensions
        n_rows = int.from_bytes(f.read(4), "big")
        n_cols = int.from_bytes(f.read(4), "big")

        # Read image data
        images = np.frombuffer(f.read(n_images * n_rows * n_cols), dtype=np.uint8)
        images = images.reshape(n_images, n_rows * n_cols)  # Flatten

        return images


def _load_labels(filepath: str) -> np.ndarray:
    """Load MNIST labels from IDX file format."""
    with open(filepath, "rb") as f:
        # Read magic number
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        # Read number of labels
        n_labels = int.from_bytes(f.read(4), "big")

        # Read label data
        labels = np.frombuffer(f.read(n_labels), dtype=np.uint8)

        return labels
