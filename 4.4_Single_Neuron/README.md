# Single Neuron Implementation

Implementation of a single artificial neuron with various activation functions and a GUI for training and visualization.

## Features

### Activation Functions
- **Heaviside** (perceptron) - Required
- **Sigmoid** (logistic) - Required
- **Sin** - For grade 4
- **Tanh** - For grade 4
- **Sign** - For grade 5
- **ReLu** - For grade 5
- **Leaky ReLu** - For grade 5

### Training
- Delta rule training: Δw_j = η * ε * f'(s) * x_j = η * (d - y) * f'(w^T * x_j) * x_j
- Training support for Heaviside and Sigmoid (required)
- Optional variable learning rate: η(n) = η_min + (η_max - η_min)(1 + cos(n/n_max * π))

### Data Generation
- Generate 2D samples from Gaussian distributions
- Multiple modes per class
- Configurable number of samples per mode
- Random means and variances

### GUI Features
- Generate and visualize 2D data samples
- Color-coded class labels (blue for class 0, red for class 1)
- Decision boundary visualization with background colors for two half-planes
- Training controls and status display
- Training error visualization over time
- Input validation and error handling

## Usage

### Running the Application

First, activate your virtual environment and install dependencies:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Then run Streamlit
streamlit run gui.py
```

Or if you prefer command line:

```bash
# Windows CMD
venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run gui.py
```

### Steps to Use

1. **Generate Data**:
   - Set number of modes per class
   - Set number of samples per mode
   - Click "Generate Data"

2. **Initialize Neuron**:
   - Select activation function
   - Set learning rate
   - Set beta (for sigmoid)
   - Click "Initialize Neuron"

3. **Train Neuron**:
   - Set number of epochs
   - Optionally enable variable learning rate
   - Set η_min and η_max (if using variable LR)
   - Click "Train Neuron"

4. **Visualize**:
   - Use the "Visualization" tab to see:
     - Blue circles: Class 0 samples
     - Red crosses: Class 1 samples
     - Light blue background: Class 0 region
     - Light coral background: Class 1 region
     - Black line: Decision boundary (only for linear activation functions)
   - Use the "Training Error" tab to see the error curve over training epochs

## File Structure

- `neuron.py` - Neuron class with all activation functions and training
- `data_generator.py` - Data generation from Gaussian distributions
- `gui.py` - Streamlit GUI application with visualization
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Requirements

- Python 3.7+
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- streamlit >= 1.28.0

## Installation

```bash
# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1  # PowerShell
# or
venv\Scripts\activate.bat    # CMD

# Install dependencies
pip install -r requirements.txt
```

## Grading Criteria

- **Grade 3**: Heaviside and Sigmoid activation functions + training + decision boundary visualization
- **Grade 4**: Grade 3 + (Sin and Tanh) OR variable learning rate
- **Grade 5**: Grade 4 + Sign, ReLu, and Leaky ReLu activation functions

## Notes

- The decision boundary is linear only for Heaviside and Sigmoid activation functions
- For non-linear activation functions (sin, tanh, ReLU, etc.), the decision boundary is non-linear and only the background coloring is shown
- For non-linear activation functions, the threshold is applied to map outputs to class labels
- Training uses the delta rule with the derivative of the activation function
- For Heaviside, the derivative is assumed to be 1 for training purposes
- The GUI includes input validation and proper error handling with logging