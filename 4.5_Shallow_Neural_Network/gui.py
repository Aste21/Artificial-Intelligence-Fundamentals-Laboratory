"""
Streamlit GUI application for training and visualizing a shallow neural network.
Includes data generation, training, and decision boundary visualization.
Supports both 2D generated data and MNIST13 dataset.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging

from neural_network import NeuralNetwork, ActivationFunction
from data_generator import DataGenerator
from mnist_loader import load_mnist

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_data_and_boundary_2d(
    samples: np.ndarray, labels: np.ndarray, network: Optional[NeuralNetwork] = None
) -> plt.Figure:
    """
    Plot 2D data samples and decision boundary.

    Args:
        samples: Input samples (n_samples, 2)
        labels: Class labels (n_samples,)
        network: Optional network for decision boundary visualization

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=80)

    # Plot data samples
    class0_mask = labels == 0
    class1_mask = labels == 1

    if np.any(class0_mask):
        ax.scatter(
            samples[class0_mask, 0],
            samples[class0_mask, 1],
            c="blue",
            marker="o",
            label="Class 0",
            alpha=0.7,
            s=20,
            edgecolors="darkblue",
            linewidths=0.5,
        )

    if np.any(class1_mask):
        ax.scatter(
            samples[class1_mask, 0],
            samples[class1_mask, 1],
            c="red",
            marker="x",
            label="Class 1",
            alpha=0.7,
            s=25,
            linewidths=1.2,
        )

    # Plot decision boundary if network is available
    if network is not None:
        try:
            _plot_decision_boundary_2d(ax, samples, network)
        except Exception as e:
            logger.warning(f"Error plotting decision boundary: {e}")

    # Set labels and title
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_title("Neural Network Visualization - 2D Data", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    return fig


def _plot_decision_boundary_2d(ax, samples: np.ndarray, network: NeuralNetwork):
    """
    Plot the decision boundary of the network for 2D data.

    Args:
        ax: matplotlib axes object
        samples: Input samples for determining plot range
        network: NeuralNetwork object
    """
    # Get the range of data
    x_min, x_max = samples[:, 0].min() - 0.5, samples[:, 0].max() + 0.5
    y_min, y_max = samples[:, 1].min() - 0.5, samples[:, 1].max() + 0.5

    # Create a mesh for background coloring (reduced resolution for smaller file size)
    resolution = 100
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions for the grid
    predictions = network.predict(grid_points)
    predictions = predictions.reshape(xx.shape)

    # Plot background with two colors for two classes
    ax.contourf(
        xx,
        yy,
        predictions,
        levels=[-0.5, 0.5, 1.5],
        colors=["#E3F2FD", "#FFEBEE"],
        alpha=0.4,
    )

    # Plot decision boundary contour
    ax.contour(
        xx,
        yy,
        predictions,
        levels=[0.5],
        colors=["black"],
        linewidths=2,
        linestyles="solid",
        alpha=0.8,
    )


def plot_training_error(training_history: list) -> plt.Figure:
    """
    Plot training error over time.

    Args:
        training_history: List of errors per epoch

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=80)

    if training_history and len(training_history) > 0:
        epochs = range(1, len(training_history) + 1)
        ax.plot(epochs, training_history, "b-", linewidth=1.5, label="MSE")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Mean Squared Error", fontsize=9)
        ax.set_title("Training Error Over Time", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add annotation for final error
        final_error = training_history[-1]
        initial_error = training_history[0]
        ax.annotate(
            f"Final: {final_error:.4f}",
            xy=(len(training_history), final_error),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=8,
        )

        # Add initial error annotation
        ax.annotate(
            f"Initial: {initial_error:.4f}",
            xy=(1, initial_error),
            xytext=(10, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            fontsize=8,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No training data available.\nTrain the network to see error plot.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Mean Squared Error", fontsize=9)
        ax.set_title("Training Error Over Time", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Shallow Neural Network Training", layout="centered")
    st.title("Shallow Neural Network Training and Visualization")

    # Initialize session state
    if "samples" not in st.session_state:
        st.session_state.samples = None
        st.session_state.labels = None
        st.session_state.network = None
        st.session_state.training_history = []
        st.session_state.data_generator = DataGenerator()
        st.session_state.dataset_type = "2D Generated"  # or "MNIST"

    # Sidebar for controls
    with st.sidebar:
        st.header("Dataset Selection")
        
        dataset_type = st.radio(
            "Choose dataset:",
            ["2D Generated", "MNIST"],
            index=0 if st.session_state.dataset_type == "2D Generated" else 1
        )
        st.session_state.dataset_type = dataset_type

        if dataset_type == "2D Generated":
            st.subheader("Data Generation")
            
            col1, col2 = st.columns(2)
            with col1:
                modes_class0 = st.number_input(
                    "Modes Class 0", min_value=1, max_value=10, value=1, step=1
                )
                samples_mode0 = st.number_input(
                    "Samples/Mode Class 0", min_value=10, max_value=500, value=50, step=10
                )

            with col2:
                modes_class1 = st.number_input(
                    "Modes Class 1", min_value=1, max_value=10, value=1, step=1
                )
                samples_mode1 = st.number_input(
                    "Samples/Mode Class 1", min_value=10, max_value=500, value=50, step=10
                )

            if st.button("Generate Data", type="primary", use_container_width=True):
                try:
                    st.session_state.samples, st.session_state.labels = (
                        st.session_state.data_generator.generate_two_class_data(
                            modes_class0, samples_mode0, modes_class1, samples_mode1
                        )
                    )
                    total_samples = len(st.session_state.samples)
                    class0_count = np.sum(st.session_state.labels == 0)
                    class1_count = np.sum(st.session_state.labels == 1)
                    st.success(
                        f"Generated {total_samples} samples (Class 0: {class0_count}, Class 1: {class1_count})"
                    )
                    st.session_state.network = None
                    st.session_state.training_history = []
                    logger.info(f"Data generated: {total_samples} samples")
                except Exception as e:
                    logger.exception("Error generating data")
                    st.error(f"Failed to generate data: {str(e)}")
        
        elif dataset_type == "MNIST":
            st.subheader("MNIST Dataset (All 10 digits)")
            
            if st.button("Load MNIST", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading MNIST dataset (all 10 digits, this may take a moment on first run)..."):
                        st.session_state.samples, st.session_state.labels = load_mnist()
                    total_samples = len(st.session_state.samples)
                    # Count samples per class
                    class_counts = {i: np.sum(st.session_state.labels == i) for i in range(10)}
                    counts_str = ", ".join([f"{i}: {class_counts[i]}" for i in range(10)])
                    st.success(
                        f"Loaded {total_samples} samples\nClasses: {counts_str}"
                    )
                    st.session_state.network = None
                    st.session_state.training_history = []
                    logger.info(f"MNIST loaded: {total_samples} samples")
                except Exception as e:
                    logger.exception("Error loading MNIST")
                    st.error(f"Failed to load MNIST: {str(e)}")
                    st.info("Make sure kagglehub is installed and you have internet connection.")

        st.divider()

        st.header("Network Configuration")
        
        # Network architecture
        st.subheader("Architecture")
        n_layers = st.number_input(
            "Number of Hidden Layers", min_value=1, max_value=3, value=1, step=1,
            help="Total layers will be: Input + Hidden + Output = 3-5 layers"
        )
        
        layer_sizes = []
        activations = []
        
        for i in range(n_layers):
            st.write(f"**Layer {i+1} (Hidden):**")
            col1, col2 = st.columns(2)
            with col1:
                size = st.number_input(
                    f"Neurons", min_value=3, max_value=100, value=10, step=1, key=f"layer_size_{i}"
                )
                layer_sizes.append(size)
            with col2:
                act = st.selectbox(
                    f"Activation", 
                    options=["logistic", "relu"],
                    index=0,
                    key=f"layer_act_{i}"
                )
                activations.append(ActivationFunction(act))
        
        # Output layer - depends on dataset
        if st.session_state.dataset_type == "2D Generated":
            output_neurons = 2
            st.write(f"**Output Layer:** {output_neurons} neurons (2 classes)")
        else:  # MNIST
            output_neurons = 10
            st.write(f"**Output Layer:** {output_neurons} neurons (10 classes: 0-9)")
        
        output_act = st.selectbox(
            "Output Activation",
            options=["logistic", "relu"],
            index=0,
            key="output_act"
        )
        activations.append(ActivationFunction(output_act))
        layer_sizes.append(output_neurons)

        learning_rate = st.slider(
            "Learning Rate (η)",
            min_value=0.001,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f",
        )

        beta = st.slider(
            "Beta (for logistic)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
        )

        if st.button("Initialize Network", type="primary", use_container_width=True):
            try:
                # Determine input dimension
                if st.session_state.dataset_type == "2D Generated":
                    input_dim = 2
                else:  # MNIST
                    input_dim = 784  # 28x28 flattened
                
                st.session_state.network = NeuralNetwork(
                    input_dim=input_dim,
                    layer_sizes=layer_sizes,
                    activations=activations,
                    learning_rate=learning_rate,
                    beta=beta,
                )
                st.session_state.training_history = []
                st.success(
                    f"Network initialized: {n_layers+1} layers (input + {n_layers} hidden + output)"
                )
                logger.info(f"Network initialized: {layer_sizes}, {[a.value for a in activations]}")
            except ValueError as e:
                logger.error(f"Validation error in network initialization: {e}")
                st.error(f"Invalid parameters: {str(e)}")
            except Exception as e:
                logger.exception("Error initializing network")
                st.error(f"Failed to initialize network: {str(e)}")

        st.divider()

        st.header("Training")

        epochs = st.number_input(
            "Epochs", min_value=1, max_value=1000, value=100, step=10
        )

        batch_size = st.number_input(
            "Batch Size", min_value=1, max_value=1000, value=32, step=1,
            help="Set to 1 for online learning, or larger for batch training"
        )

        reset_weights = st.checkbox(
            "Reset weights before training",
            value=False,
            help="If checked, weights will be reset to random values before training.",
        )

        variable_lr = st.checkbox("Variable Learning Rate", value=False)

        if variable_lr:
            col1, col2 = st.columns(2)
            with col1:
                eta_min = st.number_input(
                    "η min",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                )
            with col2:
                eta_max = st.number_input(
                    "η max",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    step=0.01,
                    format="%.2f",
                )
        else:
            eta_min = 0.01
            eta_max = 0.1

        if st.button("Train Network", type="primary", use_container_width=True):
            if st.session_state.network is None:
                st.warning("Please initialize network first")
            elif st.session_state.samples is None or st.session_state.labels is None:
                st.warning("Please load/generate data first")
            else:
                try:
                    if reset_weights:
                        st.session_state.network.reset_weights()
                        logger.info("Weights reset before training")

                    with st.spinner("Training in progress..."):
                        logger.info(
                            f"Starting training: {epochs} epochs, batch_size={batch_size}"
                        )
                        st.session_state.training_history = (
                            st.session_state.network.train(
                                st.session_state.samples,
                                st.session_state.labels,
                                epochs=epochs,
                                batch_size=batch_size if batch_size > 0 else None,
                                variable_lr=variable_lr,
                                eta_min=eta_min,
                                eta_max=eta_max,
                            )
                        )

                    initial_error = (
                        st.session_state.training_history[0]
                        if st.session_state.training_history
                        else 0
                    )
                    final_error = (
                        st.session_state.training_history[-1]
                        if st.session_state.training_history
                        else 0
                    )
                    st.success(
                        f"Training completed! Error: {initial_error:.4f} -> {final_error:.4f}"
                    )
                    logger.info(
                        f"Training completed. Initial error: {initial_error:.4f}, Final error: {final_error:.4f}"
                    )
                except Exception as e:
                    logger.exception("Error during training")
                    st.error(f"Training failed: {str(e)}")

        # Status display
        st.divider()
        st.header("Status")
        if st.session_state.samples is not None:
            total_samples = len(st.session_state.samples)
            if st.session_state.dataset_type == "2D Generated":
                class0_count = np.sum(st.session_state.labels == 0)
                class1_count = np.sum(st.session_state.labels == 1)
                st.info(
                    f"Data: {total_samples} samples (Class 0: {class0_count}, Class 1: {class1_count})"
                )
            else:  # MNIST
                class_counts = {i: np.sum(st.session_state.labels == i) for i in range(10)}
                counts_str = ", ".join([f"{i}:{class_counts[i]}" for i in range(10)])
                st.info(f"Data: {total_samples} samples\nClasses: {counts_str}")
        if st.session_state.network is not None:
            layer_info = f"{len(st.session_state.network.layer_sizes)} layers"
            st.info(
                f"Network: {layer_info} (LR={st.session_state.network.learning_rate:.3f})"
            )
        if st.session_state.training_history:
            final_error = st.session_state.training_history[-1]
            initial_error = st.session_state.training_history[0]
            st.info(
                f"Training: Final Error = {final_error:.4f} (Initial: {initial_error:.4f})"
            )

    # Main area - Visualization
    tab1, tab2 = st.tabs(["Visualization", "Training Error"])

    with tab1:
        st.header("Visualization")

        if st.session_state.samples is not None and st.session_state.labels is not None:
            if st.session_state.dataset_type == "2D Generated":
                if st.session_state.samples.shape[1] == 2:
                    fig = plot_data_and_boundary_2d(
                        st.session_state.samples,
                        st.session_state.labels,
                        st.session_state.network,
                    )
                    st.pyplot(fig, use_container_width=False)
                else:
                    st.warning("2D visualization only works for 2D data")
            else:  # MNIST
                st.info("MNIST visualization: Sample predictions for all 10 digits (0-9).")
                if st.session_state.network is not None:
                    # Show some sample predictions
                    st.subheader("Sample Predictions")
                    n_samples = min(20, len(st.session_state.samples))
                    indices = np.random.choice(len(st.session_state.samples), n_samples, replace=False)
                    sample_images = st.session_state.samples[indices]
                    sample_labels = st.session_state.labels[indices]
                    
                    predictions = st.session_state.network.predict(sample_images)
                    proba = st.session_state.network.predict_proba(sample_images)
                    
                    # Debug: check if probabilities sum to 1
                    prob_sums = np.sum(proba, axis=1)
                    if not np.allclose(prob_sums, 1.0, atol=1e-6):
                        st.warning(f"Probabilities don't sum to 1! Sums: {prob_sums[:5]}")
                    
                    # Display in grid of 5 columns
                    cols = st.columns(5)
                    for idx, (img, true_label, pred_label, prob) in enumerate(zip(
                        sample_images, sample_labels, predictions, proba
                    )):
                        with cols[idx % 5]:
                            # Reshape to 28x28 for display
                            img_2d = img.reshape(28, 28)
                            fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=80)
                            ax.imshow(img_2d, cmap='gray')
                            ax.axis('off')
                            color = 'green' if true_label == pred_label else 'red'
                            # Get confidence for predicted class
                            # Ensure pred_label is integer index
                            pred_idx = int(pred_label) if isinstance(pred_label, (np.integer, np.ndarray)) else pred_label
                            conf = float(prob[pred_idx])
                            # Clamp to [0, 1] range in case of numerical errors
                            conf = max(0.0, min(1.0, conf))
                            ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.2f}", 
                                       fontsize=7, color=color)
                            st.pyplot(fig, use_container_width=False)
                    
                    # Debug: show some probability distributions
                    if len(proba) > 0:
                        st.write("**Debug - Probability distribution for first sample:**")
                        st.write(f"Predicted class: {predictions[0]}")
                        st.write(f"Probabilities: {proba[0]}")
                        st.write(f"Sum: {np.sum(proba[0]):.6f}")
                        st.write(f"Max prob: {np.max(proba[0]):.4f} at class {np.argmax(proba[0])}")
                    
                    # Show accuracy
                    accuracy = np.mean(predictions == sample_labels)
                    st.metric("Accuracy on samples", f"{accuracy:.2%}")
                else:
                    st.info("Initialize and train network to see predictions.")
        else:
            st.info("Load/generate data to start visualization")

    with tab2:
        st.header("Training Error Over Time")

        if st.session_state.training_history:
            fig_error = plot_training_error(st.session_state.training_history)
            st.pyplot(fig_error, use_container_width=False)
        else:
            st.info("Train the network to see error history")


if __name__ == "__main__":
    main()
