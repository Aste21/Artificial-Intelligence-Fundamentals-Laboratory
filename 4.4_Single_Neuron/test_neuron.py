"""
Simple test script to verify neuron implementation.
"""

import numpy as np
from neuron import Neuron, ActivationFunction
from data_generator import DataGenerator

def test_activation_functions():
    """Test all activation functions."""
    print("Testing activation functions...")
    
    # Create test input
    x = np.array([[0.5, 0.5], [-0.5, 0.5], [0.0, 0.0]])
    
    for activation in ActivationFunction:
        print(f"\nTesting {activation.value}:")
        neuron = Neuron(input_dim=2, activation=activation)
        output = neuron.evaluate(x)
        predictions = neuron.predict(x)
        print(f"  Output: {output}")
        print(f"  Predictions: {predictions}")

def test_training():
    """Test training with Heaviside and Sigmoid."""
    print("\n\nTesting training...")
    
    # Generate simple linearly separable data
    generator = DataGenerator()
    samples, labels = generator.generate_two_class_data(
        n_modes_class0=1, n_samples_per_mode_class0=20,
        n_modes_class1=1, n_samples_per_mode_class1=20,
        seed=42
    )
    
    # Test Heaviside
    print("\nTraining with Heaviside:")
    neuron_h = Neuron(input_dim=2, activation=ActivationFunction.HEAVISIDE, learning_rate=0.1)
    history_h = neuron_h.train(samples, labels, epochs=50)
    print(f"  Final error: {history_h[-1]:.4f}")
    
    # Test Sigmoid
    print("\nTraining with Sigmoid:")
    neuron_s = Neuron(input_dim=2, activation=ActivationFunction.SIGMOID, learning_rate=0.1)
    history_s = neuron_s.train(samples, labels, epochs=50)
    print(f"  Final error: {history_s[-1]:.4f}")

def test_variable_lr():
    """Test variable learning rate."""
    print("\n\nTesting variable learning rate...")
    
    generator = DataGenerator()
    samples, labels = generator.generate_two_class_data(
        n_modes_class0=1, n_samples_per_mode_class0=20,
        n_modes_class1=1, n_samples_per_mode_class1=20,
        seed=42
    )
    
    neuron = Neuron(input_dim=2, activation=ActivationFunction.SIGMOID, learning_rate=0.1)
    history = neuron.train(samples, labels, epochs=50, variable_lr=True, eta_min=0.01, eta_max=0.1)
    print(f"  Final error: {history[-1]:.4f}")

if __name__ == "__main__":
    test_activation_functions()
    test_training()
    test_variable_lr()
    print("\n\nAll tests completed!")
