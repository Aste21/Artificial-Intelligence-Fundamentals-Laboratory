"""
Training script for Convolutional Neural Network on MNIST dataset.
"""

import numpy as np
from cnn import ConvolutionalNeuralNetwork
from mnist_loader_2d import load_mnist_2d
import time


def train_simple_cnn():
    """
    Train a simple CNN:
    - Convolutional layer (3x3, 16 filters, stride 1)
    - ReLU activation
    - Fully connected layer (10 neurons)
    - Train on MNIST
    """
    print("=" * 60)
    print("Training Simple CNN")
    print("=" * 60)
    print("Architecture:")
    print("  - Input: 28x28 greyscale images")
    print("  - Conv: 3x3, 16 filters, stride 1, padding 0")
    print("  - ReLU activation")
    print("  - Fully Connected: 10 neurons")
    print("=" * 60)

    # Load MNIST data
    print("\nLoading MNIST dataset...")
    images, labels = load_mnist_2d()
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    print(f"Labels range: {labels.min()} to {labels.max()}")

    # Use a subset for faster training (optional)
    # For full training, use all data
    # n_samples = len(images)
    n_samples = 10000  # Uncomment for faster testing

    # Split into train and test
    split_idx = int(0.8 * n_samples)
    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    test_images = images[split_idx : split_idx + min(2000, n_samples - split_idx)]
    test_labels = labels[split_idx : split_idx + min(2000, n_samples - split_idx)]

    print(f"\nTraining set: {len(train_images)} samples")
    print(f"Test set: {len(test_images)} samples")

    # Create CNN
    cnn = ConvolutionalNeuralNetwork(
        input_shape=(28, 28),
        conv_layers=[
            {"num_filters": 16, "filter_size": (3, 3), "stride": 1, "padding": 0}
        ],
        pool_layers=None,  # No pooling
        fc_layers=[10],  # Single FC layer with 10 neurons
        learning_rate=0.01,
    )

    print("\nNetwork created successfully!")

    # Train
    print("\nStarting training...")
    start_time = time.time()
    losses = cnn.train(
        train_images, train_labels, epochs=3, batch_size=64, verbose=True
    )
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Evaluate
    print("\nEvaluating on test set...")
    predictions = cnn.predict(test_images)
    accuracy = np.mean(predictions == test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return cnn, losses, accuracy


def train_cnn_with_pooling():
    """
    Train a CNN with pooling:
    - Convolutional layer (3x3, 16 filters, stride 1)
    - ReLU activation
    - Max-pooling layer (2x2, stride 2)
    - Fully connected layer (10 neurons)
    - Softmax activation
    - Train on MNIST
    """
    print("\n" + "=" * 60)
    print("Training CNN with Pooling")
    print("=" * 60)
    print("Architecture:")
    print("  - Input: 28x28 greyscale images")
    print("  - Conv: 3x3, 16 filters, stride 1, padding 0")
    print("  - ReLU activation")
    print("  - Max-pooling: 2x2, stride 2")
    print("  - Fully Connected: 10 neurons")
    print("  - Softmax activation")
    print("=" * 60)

    # Load MNIST data
    print("\nLoading MNIST dataset...")
    images, labels = load_mnist_2d()
    print(f"Loaded {len(images)} images")

    # Use a subset for faster training (optional)
    # n_samples = len(images)
    n_samples = 10000  # Uncomment for faster testing

    # Split into train and test
    split_idx = int(0.8 * n_samples)
    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    test_images = images[split_idx : split_idx + min(2000, n_samples - split_idx)]
    test_labels = labels[split_idx : split_idx + min(2000, n_samples - split_idx)]

    print(f"\nTraining set: {len(train_images)} samples")
    print(f"Test set: {len(test_images)} samples")

    # Create CNN with pooling
    cnn = ConvolutionalNeuralNetwork(
        input_shape=(28, 28),
        conv_layers=[
            {"num_filters": 16, "filter_size": (3, 3), "stride": 1, "padding": 0}
        ],
        pool_layers=[{"pool_size": (2, 2), "stride": 2}],
        fc_layers=[10],  # Single FC layer with 10 neurons
        learning_rate=0.01,
    )

    print("\nNetwork created successfully!")

    # Train
    print("\nStarting training...")
    start_time = time.time()
    losses = cnn.train(
        train_images, train_labels, epochs=3, batch_size=64, verbose=True
    )
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Evaluate
    print("\nEvaluating on test set...")
    predictions = cnn.predict(test_images)
    accuracy = np.mean(predictions == test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return cnn, losses, accuracy


if __name__ == "__main__":
    print("Convolutional Neural Network Training")
    print("=" * 60)

    # Train simple CNN
    try:
        cnn1, losses1, acc1 = train_simple_cnn()
        print(f"\n✓ Simple CNN training completed. Accuracy: {acc1 * 100:.2f}%")
    except Exception as e:
        print(f"\n✗ Simple CNN training failed: {e}")
        import traceback

        traceback.print_exc()

    # Train CNN with pooling
    try:
        cnn2, losses2, acc2 = train_cnn_with_pooling()
        print(f"\n✓ CNN with pooling training completed. Accuracy: {acc2 * 100:.2f}%")
    except Exception as e:
        print(f"\n✗ CNN with pooling training failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Training complete!")
