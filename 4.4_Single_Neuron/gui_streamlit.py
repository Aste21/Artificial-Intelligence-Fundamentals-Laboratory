"""
Streamlit GUI application for training and visualizing a single neuron.
Includes data generation, training, and decision boundary visualization.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from neuron import Neuron, ActivationFunction
from data_generator import DataGenerator


def plot_data_and_boundary(samples: np.ndarray, labels: np.ndarray, 
                          neuron: Optional[Neuron] = None):
    """Plot data samples and decision boundary."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot data samples
    class0_mask = labels == 0
    class1_mask = labels == 1
    
    if np.any(class0_mask):
        ax.scatter(samples[class0_mask, 0], samples[class0_mask, 1],
                  c='blue', marker='o', label='Class 0', alpha=0.6, s=50)
    
    if np.any(class1_mask):
        ax.scatter(samples[class1_mask, 0], samples[class1_mask, 1],
                  c='red', marker='x', label='Class 1', alpha=0.6, s=50)
    
    # Plot decision boundary if neuron is available
    if neuron is not None:
        try:
            # Get the range of data
            x_min, x_max = samples[:, 0].min() - 0.5, samples[:, 0].max() + 0.5
            y_min, y_max = samples[:, 1].min() - 0.5, samples[:, 1].max() + 0.5
            
            # Create a mesh for background coloring
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                               np.linspace(y_min, y_max, 200))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            # Get continuous outputs for the grid
            outputs = neuron.evaluate(grid_points)
            outputs = outputs.reshape(xx.shape)
            
            # For classification, threshold at 0.5
            predictions = (outputs >= 0.5).astype(float)
            
            # Plot background with two colors for two half-planes
            ax.contourf(xx, yy, predictions, levels=[0, 0.5, 1], 
                       colors=['lightblue', 'lightcoral'], alpha=0.3)
            
            # Plot decision boundary line
            try:
                a, b, c = neuron.get_decision_boundary_params()
                # Decision boundary: a*x + b*y + c = 0
                if abs(b) > 1e-10:
                    x_line = np.linspace(x_min, x_max, 100)
                    y_line = (-a * x_line - c) / b
                    mask = (y_line >= y_min) & (y_line <= y_max)
                    if np.any(mask):
                        ax.plot(x_line[mask], y_line[mask], 'k--', linewidth=2, label='Decision Boundary')
                else:
                    x_line = -c / a if abs(a) > 1e-10 else 0
                    ax.axvline(x=x_line, color='k', linestyle='--', linewidth=2, label='Decision Boundary')
            except:
                pass
        except Exception as e:
            st.warning(f"Could not plot decision boundary: {e}")
    
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title("Neuron Training Visualization", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Single Neuron Training", layout="wide")
    st.title("🧠 Single Neuron Training and Visualization")
    
    # Initialize session state
    if 'samples' not in st.session_state:
        st.session_state.samples = None
        st.session_state.labels = None
        st.session_state.neuron = None
        st.session_state.training_history = []
        st.session_state.data_generator = DataGenerator()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📊 Data Generation")
        
        col1, col2 = st.columns(2)
        with col1:
            modes_class0 = st.number_input("Modes Class 0", min_value=1, max_value=10, value=1, step=1)
            samples_mode0 = st.number_input("Samples/Mode Class 0", min_value=10, max_value=500, value=50, step=10)
        
        with col2:
            modes_class1 = st.number_input("Modes Class 1", min_value=1, max_value=10, value=1, step=1)
            samples_mode1 = st.number_input("Samples/Mode Class 1", min_value=10, max_value=500, value=50, step=10)
        
        if st.button("🔄 Generate Data", type="primary", use_container_width=True):
            try:
                st.session_state.samples, st.session_state.labels = st.session_state.data_generator.generate_two_class_data(
                    modes_class0, samples_mode0, modes_class1, samples_mode1
                )
                st.success(f"✅ Generated {len(st.session_state.samples)} samples")
                st.session_state.neuron = None  # Reset neuron when new data is generated
            except Exception as e:
                st.error(f"Failed to generate data: {str(e)}")
        
        st.divider()
        
        st.header("⚙️ Neuron Configuration")
        
        activation_name = st.selectbox(
            "Activation Function",
            options=["heaviside", "sigmoid", "sin", "tanh", "sign", "relu", "leaky_relu"],
            index=1
        )
        
        learning_rate = st.slider("Learning Rate (η)", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
        
        beta = st.slider("Beta (for sigmoid)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, format="%.1f")
        
        if st.button("🔧 Initialize Neuron", type="primary", use_container_width=True):
            try:
                activation = ActivationFunction(activation_name)
                st.session_state.neuron = Neuron(
                    input_dim=2, 
                    activation=activation, 
                    learning_rate=learning_rate, 
                    beta=beta
                )
                st.success(f"✅ Neuron initialized with {activation_name}")
            except Exception as e:
                st.error(f"Failed to initialize neuron: {str(e)}")
        
        st.divider()
        
        st.header("🎓 Training")
        
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100, step=10)
        
        variable_lr = st.checkbox("Variable Learning Rate", value=False)
        
        if variable_lr:
            col1, col2 = st.columns(2)
            with col1:
                eta_min = st.number_input("η min", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
            with col2:
                eta_max = st.number_input("η max", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
        else:
            eta_min = 0.01
            eta_max = 0.1
        
        if st.button("🚀 Train Neuron", type="primary", use_container_width=True):
            if st.session_state.neuron is None:
                st.warning("⚠️ Please initialize neuron first")
            elif st.session_state.samples is None or st.session_state.labels is None:
                st.warning("⚠️ Please generate data first")
            else:
                try:
                    with st.spinner("Training in progress..."):
                        st.session_state.training_history = st.session_state.neuron.train(
                            st.session_state.samples, 
                            st.session_state.labels, 
                            epochs=epochs,
                            variable_lr=variable_lr, 
                            eta_min=eta_min, 
                            eta_max=eta_max
                        )
                    final_error = st.session_state.training_history[-1]
                    st.success(f"✅ Training completed! Final error: {final_error:.4f}")
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
        
        # Status display
        st.divider()
        st.header("📈 Status")
        if st.session_state.samples is not None:
            st.info(f"📊 Data: {len(st.session_state.samples)} samples")
        if st.session_state.neuron is not None:
            st.info(f"🧠 Neuron: {st.session_state.neuron.activation_type.value}")
        if st.session_state.training_history:
            st.info(f"📉 Final Error: {st.session_state.training_history[-1]:.4f}")
    
    # Main area - Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Visualization")
        
        if st.session_state.samples is not None and st.session_state.labels is not None:
            fig = plot_data_and_boundary(
                st.session_state.samples, 
                st.session_state.labels, 
                st.session_state.neuron
            )
            st.pyplot(fig)
        else:
            st.info("👈 Generate data to start visualization")
    
    with col2:
        st.header("📉 Training History")
        
        if st.session_state.training_history:
            fig_history, ax_history = plt.subplots(figsize=(6, 4))
            ax_history.plot(st.session_state.training_history)
            ax_history.set_xlabel("Epoch")
            ax_history.set_ylabel("Error")
            ax_history.set_title("Training Error Over Time")
            ax_history.grid(True, alpha=0.3)
            st.pyplot(fig_history)
        else:
            st.info("Train the neuron to see error history")


if __name__ == "__main__":
    main()
