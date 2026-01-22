"""
GUI application for training and visualizing a single neuron.
Includes data generation, training, and decision boundary visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import Optional, Tuple

from neuron import Neuron, ActivationFunction
from data_generator import DataGenerator


class NeuronGUI:
    """GUI application for neuron training and visualization."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Single Neuron Training and Visualization")
        self.root.geometry("1200x800")
        
        # Data storage
        self.samples: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.neuron: Optional[Neuron] = None
        self.training_history: list = []
        
        # Data generator
        self.data_generator = DataGenerator()
        
        # Create GUI elements
        self._create_widgets()
        
        # Initialize plot
        self._update_plot()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Data generation section
        data_frame = ttk.LabelFrame(control_frame, text="Data Generation", padding="5")
        data_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(data_frame, text="Modes Class 0:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.modes_class0_var = tk.IntVar(value=1)
        ttk.Spinbox(data_frame, from_=1, to=10, textvariable=self.modes_class0_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(data_frame, text="Samples/Mode Class 0:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.samples_mode0_var = tk.IntVar(value=50)
        ttk.Spinbox(data_frame, from_=10, to=500, textvariable=self.samples_mode0_var, width=10).grid(row=1, column=1, pady=2)
        
        ttk.Label(data_frame, text="Modes Class 1:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.modes_class1_var = tk.IntVar(value=1)
        ttk.Spinbox(data_frame, from_=1, to=10, textvariable=self.modes_class1_var, width=10).grid(row=2, column=1, pady=2)
        
        ttk.Label(data_frame, text="Samples/Mode Class 1:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.samples_mode1_var = tk.IntVar(value=50)
        ttk.Spinbox(data_frame, from_=10, to=500, textvariable=self.samples_mode1_var, width=10).grid(row=3, column=1, pady=2)
        
        ttk.Button(data_frame, text="Generate Data", command=self._generate_data).grid(row=4, column=0, columnspan=2, pady=5)
        
        # Neuron configuration section
        neuron_frame = ttk.LabelFrame(control_frame, text="Neuron Configuration", padding="5")
        neuron_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(neuron_frame, text="Activation Function:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.activation_var = tk.StringVar(value="sigmoid")
        activation_combo = ttk.Combobox(neuron_frame, textvariable=self.activation_var, 
                                       values=["heaviside", "sigmoid", "sin", "tanh", "sign", "relu", "leaky_relu"],
                                       state="readonly", width=15)
        activation_combo.grid(row=0, column=1, pady=2)
        
        ttk.Label(neuron_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(neuron_frame, from_=0.001, to=1.0, increment=0.01, textvariable=self.lr_var, width=10).grid(row=1, column=1, pady=2)
        
        ttk.Label(neuron_frame, text="Beta (for sigmoid):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.beta_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(neuron_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.beta_var, width=10).grid(row=2, column=1, pady=2)
        
        ttk.Button(neuron_frame, text="Initialize Neuron", command=self._initialize_neuron).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Training section
        training_frame = ttk.LabelFrame(control_frame, text="Training", padding="5")
        training_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(training_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(training_frame, from_=1, to=1000, textvariable=self.epochs_var, width=10).grid(row=0, column=1, pady=2)
        
        self.variable_lr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(training_frame, text="Variable Learning Rate", variable=self.variable_lr_var).grid(row=1, column=0, columnspan=2, pady=2)
        
        ttk.Label(training_frame, text="η min:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.eta_min_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(training_frame, from_=0.001, to=0.1, increment=0.001, textvariable=self.eta_min_var, width=10).grid(row=2, column=1, pady=2)
        
        ttk.Label(training_frame, text="η max:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.eta_max_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(training_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.eta_max_var, width=10).grid(row=3, column=1, pady=2)
        
        ttk.Button(training_frame, text="Train Neuron", command=self._train_neuron).grid(row=4, column=0, columnspan=2, pady=5)
        
        # Status section
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="5")
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", wraplength=200)
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Right panel - Plot
        plot_frame = ttk.Frame(main_frame, padding="5")
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
    
    def _generate_data(self):
        """Generate new data samples."""
        try:
            n_modes0 = self.modes_class0_var.get()
            n_samples0 = self.samples_mode0_var.get()
            n_modes1 = self.modes_class1_var.get()
            n_samples1 = self.samples_mode1_var.get()
            
            self.samples, self.labels = self.data_generator.generate_two_class_data(
                n_modes0, n_samples0, n_modes1, n_samples1
            )
            
            self.status_label.config(text=f"Generated {len(self.samples)} samples")
            self._update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
    
    def _initialize_neuron(self):
        """Initialize a new neuron with specified parameters."""
        try:
            activation_name = self.activation_var.get()
            activation = ActivationFunction(activation_name)
            learning_rate = self.lr_var.get()
            beta = self.beta_var.get()
            
            # Neuron needs 2 inputs for 2D data
            self.neuron = Neuron(input_dim=2, activation=activation, 
                                learning_rate=learning_rate, beta=beta)
            
            self.status_label.config(text=f"Neuron initialized with {activation_name}")
            self._update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize neuron: {str(e)}")
    
    def _train_neuron(self):
        """Train the neuron on the generated data."""
        if self.neuron is None:
            messagebox.showwarning("Warning", "Please initialize neuron first")
            return
        
        if self.samples is None or self.labels is None:
            messagebox.showwarning("Warning", "Please generate data first")
            return
        
        try:
            epochs = self.epochs_var.get()
            variable_lr = self.variable_lr_var.get()
            eta_min = self.eta_min_var.get()
            eta_max = self.eta_max_var.get()
            
            self.training_history = self.neuron.train(
                self.samples, self.labels, epochs=epochs,
                variable_lr=variable_lr, eta_min=eta_min, eta_max=eta_max
            )
            
            self.status_label.config(text=f"Training completed. Final error: {self.training_history[-1]:.4f}")
            self._update_plot()
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def _update_plot(self):
        """Update the visualization plot."""
        self.ax.clear()
        
        # Plot data samples if available
        if self.samples is not None and self.labels is not None:
            # Class 0 samples (blue)
            class0_mask = self.labels == 0
            if np.any(class0_mask):
                self.ax.scatter(self.samples[class0_mask, 0], self.samples[class0_mask, 1],
                              c='blue', marker='o', label='Class 0', alpha=0.6, s=30)
            
            # Class 1 samples (red)
            class1_mask = self.labels == 1
            if np.any(class1_mask):
                self.ax.scatter(self.samples[class1_mask, 0], self.samples[class1_mask, 1],
                              c='red', marker='x', label='Class 1', alpha=0.6, s=30)
        
        # Plot decision boundary if neuron is trained
        if self.neuron is not None:
            try:
                self._plot_decision_boundary()
            except Exception as e:
                # If decision boundary plotting fails, continue without it
                pass
        
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Neuron Training Visualization")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal', adjustable='box')
        
        self.canvas.draw()
    
    def _plot_decision_boundary(self):
        """Plot the decision boundary of the neuron."""
        if self.samples is None:
            return
        
        # Get the range of data
        x_min, x_max = self.samples[:, 0].min() - 0.5, self.samples[:, 0].max() + 0.5
        y_min, y_max = self.samples[:, 1].min() - 0.5, self.samples[:, 1].max() + 0.5
        
        # Create a mesh for background coloring
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                           np.linspace(y_min, y_max, 200))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Get continuous outputs for the grid (for better visualization)
        outputs = self.neuron.evaluate(grid_points)
        outputs = outputs.reshape(xx.shape)
        
        # For classification, threshold at 0.5
        # But use continuous values for smoother visualization
        predictions = (outputs >= 0.5).astype(float)
        
        # Plot background with two colors for two half-planes
        self.ax.contourf(xx, yy, predictions, levels=[0, 0.5, 1], 
                        colors=['lightblue', 'lightcoral'], alpha=0.3)
        
        # Plot decision boundary line
        try:
            a, b, c = self.neuron.get_decision_boundary_params()
            # Decision boundary: a*x + b*y + c = 0
            # Rearrange: y = (-a*x - c) / b
            if abs(b) > 1e-10:
                x_line = np.linspace(x_min, x_max, 100)
                y_line = (-a * x_line - c) / b
                # Filter y values within plot range
                mask = (y_line >= y_min) & (y_line <= y_max)
                if np.any(mask):
                    self.ax.plot(x_line[mask], y_line[mask], 'k--', linewidth=2, label='Decision Boundary')
            else:
                # Vertical line: x = -c/a
                x_line = -c / a if abs(a) > 1e-10 else 0
                self.ax.axvline(x=x_line, color='k', linestyle='--', linewidth=2, label='Decision Boundary')
        except:
            # If boundary calculation fails, just show the background
            pass


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = NeuronGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
