"""
Data generator for creating 2D samples from Gaussian distributions.
Each class can have multiple Gaussian modes.
Reused from task 4.4.
"""

import numpy as np
from typing import List, Tuple, Optional


class DataGenerator:
    """Generator for 2D data samples from Gaussian distributions."""
    
    def __init__(self, mu_range: Tuple[float, float] = (-1.0, 1.0), 
                 sigma_range: Tuple[float, float] = (0.1, 0.3)):
        """
        Initialize the data generator.
        
        Args:
            mu_range: Range for mean values (μx, μy)
            sigma_range: Range for standard deviation values
        """
        self.mu_range = mu_range
        self.sigma_range = sigma_range
    
    def generate_class_data(self, n_modes: int, n_samples_per_mode: int, 
                           class_label: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data samples for one class with multiple Gaussian modes.
        
        Args:
            n_modes: Number of Gaussian modes for this class
            n_samples_per_mode: Number of samples per mode
            class_label: Class label (0 or 1)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (samples, labels) where samples is (n_samples, 2) and labels is (n_samples,)
        """
        if seed is not None:
            np.random.seed(seed)
        
        all_samples = []
        all_labels = []
        
        for _ in range(n_modes):
            # Random mean in the specified range
            mu_x = np.random.uniform(self.mu_range[0], self.mu_range[1])
            mu_y = np.random.uniform(self.mu_range[0], self.mu_range[1])
            
            # Random standard deviation
            sigma_x = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            sigma_y = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            
            # Generate samples from 2D Gaussian
            samples = np.random.normal(
                loc=[mu_x, mu_y],
                scale=[sigma_x, sigma_y],
                size=(n_samples_per_mode, 2)
            )
            
            all_samples.append(samples)
            all_labels.extend([class_label] * n_samples_per_mode)
        
        # Concatenate all samples
        samples_array = np.vstack(all_samples)
        labels_array = np.array(all_labels)
        
        return samples_array, labels_array
    
    def generate_two_class_data(self, n_modes_class0: int, n_samples_per_mode_class0: int,
                                n_modes_class1: int, n_samples_per_mode_class1: int,
                                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for two classes.
        
        Args:
            n_modes_class0: Number of modes for class 0
            n_samples_per_mode_class0: Samples per mode for class 0
            n_modes_class1: Number of modes for class 1
            n_samples_per_mode_class1: Samples per mode for class 1
            seed: Random seed
            
        Returns:
            Tuple of (samples, labels)
        """
        # Generate class 0
        x0, y0 = self.generate_class_data(n_modes_class0, n_samples_per_mode_class0, 0, seed)
        
        # Generate class 1 with different seed offset
        seed1 = seed + 1000 if seed is not None else None
        x1, y1 = self.generate_class_data(n_modes_class1, n_samples_per_mode_class1, 1, seed1)
        
        # Combine
        samples = np.vstack([x0, x1])
        labels = np.hstack([y0, y1])
        
        return samples, labels
