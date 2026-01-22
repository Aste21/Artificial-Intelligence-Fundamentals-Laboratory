"""
Streamlit GUI application for training and visualizing a single neuron.
Includes data generation, training, and decision boundary visualization.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import logging

from neuron import Neuron, ActivationFunction
from data_generator import DataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_data_and_boundary(
    samples: np.ndarray, labels: np.ndarray, neuron: Optional[Neuron] = None
) -> plt.Figure:
    """
    Plot data samples and decision boundary.

    Args:
        samples: Input samples (n_samples, 2)
        labels: Class labels (n_samples,)
        neuron: Optional neuron for decision boundary visualization

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

        # Plot decision boundary if neuron is available
        if neuron is not None:
            try:
                _plot_decision_boundary(ax, samples, neuron)
            except Exception as e:
                logger.warning(f"Error plotting decision boundary: {e}")

    # Set labels and title
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)

    if neuron is not None:
        activation_name = neuron.evaluation_activation.value
        title = f"Neuron Visualization - {activation_name.capitalize()}"
    else:
        title = "Neuron Visualization - No neuron initialized"

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")

    # Tight layout to reduce padding
    plt.tight_layout()

    return fig


def _plot_decision_boundary(ax, samples: np.ndarray, neuron: Neuron):
    """
    Plot the decision boundary of the neuron.

    Args:
        ax: matplotlib axes object
        samples: Input samples for determining plot range
        neuron: Neuron object
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

    # Get continuous outputs for the grid
    outputs = neuron.evaluate(grid_points)
    outputs = outputs.reshape(xx.shape)

    # Determine threshold based on evaluation activation function
    if neuron.evaluation_activation == ActivationFunction.HEAVISIDE:
        threshold = 0.5
    elif neuron.evaluation_activation == ActivationFunction.SIGMOID:
        threshold = 0.5
    elif neuron.evaluation_activation == ActivationFunction.TANH:
        threshold = 0.0
    elif neuron.evaluation_activation == ActivationFunction.SIN:
        threshold = 0.0
    elif neuron.evaluation_activation == ActivationFunction.SIGN:
        threshold = 0.0
    elif neuron.evaluation_activation == ActivationFunction.RELU:
        threshold = 0.0
    elif neuron.evaluation_activation == ActivationFunction.LEAKY_RELU:
        threshold = 0.0
    else:
        threshold = 0.5

    # Create predictions for background coloring
    predictions = (outputs >= threshold).astype(float)

    # Plot background with two colors for two half-planes
    ax.contourf(
        xx,
        yy,
        predictions,
        levels=[0, 0.5, 1],
        colors=["#E3F2FD", "#FFEBEE"],
        alpha=0.4,
    )

    # Plot decision boundary line
    # For a single neuron, the decision boundary is always linear
    try:
        a, b, c = neuron.get_decision_boundary_params()
        # Decision boundary: a*x + b*y + c = 0
        # Rearrange: y = (-a*x - c) / b
        if abs(b) > 1e-10:
            x_line = np.linspace(x_min, x_max, 200)
            y_line = (-a * x_line - c) / b
            # Filter y values within plot range
            mask = (y_line >= y_min) & (y_line <= y_max)
            if np.any(mask):
                ax.plot(
                    x_line[mask],
                    y_line[mask],
                    "k-",
                    linewidth=2,
                    label="Decision Boundary",
                    alpha=0.8,
                )
        else:
            # Vertical line: x = -c/a
            if abs(a) > 1e-10:
                x_line = -c / a
                ax.axvline(
                    x=x_line,
                    color="k",
                    linestyle="-",
                    linewidth=2,
                    label="Decision Boundary",
                    alpha=0.8,
                )
    except Exception as e:
        logger.warning(f"Could not plot decision boundary: {e}")


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
            "No training data available.\nTrain the neuron to see error plot.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Mean Squared Error", fontsize=9)
        ax.set_title("Training Error Over Time", fontsize=10, fontweight="bold")

    # Tight layout to reduce padding
    plt.tight_layout()

    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Single Neuron Training", layout="centered")
    st.title("Single Neuron Training and Visualization")

    # Initialize session state
    if "samples" not in st.session_state:
        st.session_state.samples = None
        st.session_state.labels = None
        st.session_state.neuron = None
        st.session_state.training_history = []
        st.session_state.data_generator = DataGenerator()

    # Sidebar for controls
    with st.sidebar:
        st.header("Data Generation")

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
                # Validate inputs
                if modes_class0 < 1 or modes_class1 < 1:
                    st.error("Number of modes must be at least 1")
                elif samples_mode0 < 10 or samples_mode1 < 10:
                    st.error("Number of samples per mode must be at least 10")
                else:
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
                    st.session_state.neuron = (
                        None  # Reset neuron when new data is generated
                    )
                    st.session_state.training_history = []
                    logger.info(f"Data generated: {total_samples} samples")
            except ValueError as e:
                logger.error(f"Validation error in data generation: {e}")
                st.error(f"Invalid input parameters: {str(e)}")
            except Exception as e:
                logger.exception("Error generating data")
                st.error(f"Failed to generate data: {str(e)}")

        st.divider()

        st.header("Neuron Configuration")

        training_activation_name = st.selectbox(
            "Training Activation Function",
            options=["heaviside", "sigmoid", "sin", "tanh"],
            index=1,
            help="Activation function used during training. Only Heaviside, Sigmoid, Sin, and Tanh are supported for training.",
        )

        evaluation_activation_name = st.selectbox(
            "Evaluation Activation Function",
            options=[
                "heaviside",
                "sigmoid",
                "sin",
                "tanh",
                "sign",
                "relu",
                "leaky_relu",
            ],
            index=1,
            help="Activation function used for evaluation and prediction. All functions are supported.",
        )

        learning_rate = st.slider(
            "Learning Rate (η)",
            min_value=0.001,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f",
        )

        beta = st.slider(
            "Beta (for sigmoid)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
        )

        if st.button("Initialize Neuron", type="primary", use_container_width=True):
            try:
                # Validate parameters
                if learning_rate <= 0 or learning_rate > 1.0:
                    st.error("Learning rate must be between 0.001 and 1.0")
                elif beta <= 0 or beta > 10.0:
                    st.error("Beta must be between 0.1 and 10.0")
                else:
                    training_activation = ActivationFunction(training_activation_name)
                    evaluation_activation = ActivationFunction(
                        evaluation_activation_name
                    )
                    st.session_state.neuron = Neuron(
                        input_dim=2,
                        training_activation=training_activation,
                        evaluation_activation=evaluation_activation,
                        learning_rate=learning_rate,
                        beta=beta,
                    )
                    st.session_state.training_history = []
                    st.success(
                        f"Neuron initialized: Training={training_activation_name}, Evaluation={evaluation_activation_name} (LR={learning_rate:.3f}, beta={beta:.1f})"
                    )
                    logger.info(
                        f"Neuron initialized: Training={training_activation_name}, Evaluation={evaluation_activation_name}, LR={learning_rate}, beta={beta}"
                    )
            except ValueError as e:
                logger.error(f"Validation error in neuron initialization: {e}")
                st.error(f"Invalid parameters: {str(e)}")
            except Exception as e:
                logger.exception("Error initializing neuron")
                st.error(f"Failed to initialize neuron: {str(e)}")

        st.divider()

        st.header("Training")

        epochs = st.number_input(
            "Epochs", min_value=1, max_value=1000, value=100, step=10
        )

        reset_weights = st.checkbox(
            "Reset weights before training",
            value=False,
            help="If checked, weights will be reset to random values before training. Otherwise, training continues from current weights.",
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

        if st.button("Train Neuron", type="primary", use_container_width=True):
            # Validation
            if st.session_state.neuron is None:
                st.warning("Please initialize neuron first")
            elif st.session_state.samples is None or st.session_state.labels is None:
                st.warning("Please generate data first")
            elif len(st.session_state.samples) == 0:
                st.warning("No data samples available")
            else:
                try:
                    # Validate training parameters
                    if epochs < 1 or epochs > 1000:
                        st.error("Epochs must be between 1 and 1000")
                    elif variable_lr and (
                        eta_min >= eta_max or eta_min <= 0 or eta_max <= 0
                    ):
                        st.error(
                            "Eta min must be less than eta max and both must be positive"
                        )
                    elif st.session_state.samples.shape[1] != 2:
                        st.error("Data must be 2D for visualization")
                    elif len(st.session_state.labels) != len(st.session_state.samples):
                        st.error("Number of labels does not match number of samples")
                    else:
                        # Reset weights if requested
                        if reset_weights:
                            st.session_state.neuron.reset_weights()
                            logger.info("Weights reset before training")

                        with st.spinner("Training in progress..."):
                            logger.info(
                                f"Starting training: {epochs} epochs, variable_lr={variable_lr}, reset_weights={reset_weights}"
                            )
                            st.session_state.training_history = (
                                st.session_state.neuron.train(
                                    st.session_state.samples,
                                    st.session_state.labels,
                                    epochs=epochs,
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
                except ValueError as e:
                    logger.error(f"Validation error in training: {e}")
                    st.error(f"Invalid parameters: {str(e)}")
                except Exception as e:
                    logger.exception("Error during training")
                    st.error(f"Training failed: {str(e)}")

        # Status display
        st.divider()
        st.header("Status")
        if st.session_state.samples is not None:
            total_samples = len(st.session_state.samples)
            class0_count = np.sum(st.session_state.labels == 0)
            class1_count = np.sum(st.session_state.labels == 1)
            st.info(
                f"Data: {total_samples} samples (Class 0: {class0_count}, Class 1: {class1_count})"
            )
        if st.session_state.neuron is not None:
            training_name = st.session_state.neuron.training_activation.value
            evaluation_name = st.session_state.neuron.evaluation_activation.value
            st.info(
                f"Neuron: Training={training_name}, Eval={evaluation_name} (LR={st.session_state.neuron.learning_rate:.3f})"
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
            fig = plot_data_and_boundary(
                st.session_state.samples,
                st.session_state.labels,
                st.session_state.neuron,
            )
            st.pyplot(fig, use_container_width=False)
        else:
            st.info("Generate data to start visualization")

    with tab2:
        st.header("Training Error Over Time")

        if st.session_state.training_history:
            fig_error = plot_training_error(st.session_state.training_history)
            st.pyplot(fig_error, use_container_width=False)
        else:
            st.info("Train the neuron to see error history")


if __name__ == "__main__":
    main()
