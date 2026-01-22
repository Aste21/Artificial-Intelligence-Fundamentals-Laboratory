# Shallow Neural Network Implementation

Implementation of a shallow fully connected neural network (up to 5 layers) with configurable architecture, logistic and ReLU activation functions, and training using backpropagation.

## Features

### Network Architecture
- **Configurable layers**: Up to 5 layers total (input + hidden + output)
- **Minimum 3 layers**: Input layer (2 inputs, multiple neurons >2), at least one hidden layer, output layer (2 neurons)
- **Flexible architecture**: Configurable number of hidden layers and neurons per layer
- **Activation functions**: Logistic (sigmoid) and ReLU
- **Backpropagation training**: Implements the formula:
  ```
  Δw_k^j = η * δ_k * f'(s_k) * x_l = η * Σ_i(d_i - y_i) * f'(s_i) * W_i^k * f'(s_k) * x_l
  ```

### Training
- **Backpropagation**: Full backpropagation algorithm for multi-layer networks
- **Batch training**: Support for batch training (configurable batch size)
- **Variable learning rate**: Optional variable learning rate scheduling
- **Multiple datasets**: Support for 2D generated data and MNIST13 dataset

### Data Support
- **2D Generated Data**: Uses data generator from task 4.4 (Gaussian distributions with multiple modes per class)
- **MNIST13**: MNIST dataset filtered for digits 1 and 3 (flattened pixel vectors)

### GUI Features
- **Streamlit interface**: Modern web-based GUI similar to task 4.4
- **Decision boundary visualization**: Color-coded regions for 2D data classification
- **Training visualization**: Real-time training error plots
- **Sample predictions**: For MNIST13, shows sample images with predictions
- **Configurable network**: Easy-to-use interface for setting network architecture

## Installation

### Prerequisites
- Python 3.7 or higher
- Internet connection (for downloading MNIST dataset on first run)

### Setup - Krok po kroku (Windows CMD)

#### 1. Otwórz CMD (Command Prompt) i przejdź do folderu projektu:
```cmd
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.5_Shallow_Neural_Network
```

#### 2. Stwórz środowisko wirtualne (venv):
```cmd
python -m venv venv
```

#### 3. Aktywuj środowisko wirtualne:
```cmd
venv\Scripts\activate.bat
```

**Po aktywacji powinieneś zobaczyć `(venv)` na początku linii komend.**

#### 4. Zainstaluj zależności:
```cmd
pip install -r requirements.txt
```

Lub jeśli pip nie działa:
```cmd
python -m pip install -r requirements.txt
```

#### 5. Sprawdź czy wszystko działa (opcjonalnie):
```cmd
python test_network.py
```

#### 6. Uruchom aplikację Streamlit:
```cmd
streamlit run gui.py --server.port 9001
```

Aplikacja automatycznie otworzy się w przeglądarce na `http://localhost:9001`

---

### Szybkie komendy (kopiuj-wklej dla CMD):

```cmd
cd C:\Users\kopcz\Documents\GitHub\Artificial-Intelligence-Fundamentals-Laboratory\4.5_Shallow_Neural_Network
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run gui.py --server.port 9001
```

**Uwaga:** Po każdej komendzie naciśnij Enter. Po aktywacji venv zobaczysz `(venv)` na początku linii.

### Deaktywacja środowiska wirtualnego:
Gdy skończysz pracę, możesz deaktywować venv:
```powershell
deactivate
```

## Usage

### Running the Application

**Start the Streamlit GUI on port 9001:**
```bash
streamlit run gui.py --server.port 9001
```

The application will open in your default web browser at `http://localhost:9001`.

**Alternative (if the above doesn't work):**
```bash
streamlit run gui.py --server.port 9001 --server.address localhost
```

Or create a `.streamlit/config.toml` file in the project directory:
```toml
[server]
port = 9001
```

### Using the GUI

#### 1. Select Dataset

**Option A: 2D Generated Data**
- Choose "2D Generated" dataset
- Set number of modes per class (1-10)
- Set number of samples per mode (10-500)
- Click "🔄 Generate Data"

**Option B: MNIST13 Dataset**
- Choose "MNIST13" dataset
- Click "📥 Load MNIST13"
- The dataset will be automatically downloaded on first run (requires internet connection)
- Note: This may take a few minutes on first run

#### 2. Configure Network

- **Number of Hidden Layers**: Set to 1-3 (total layers will be 3-5)
- **Layer Configuration**: For each hidden layer:
  - Set number of neurons (minimum 3 for 2D input)
  - Choose activation function (logistic or ReLU)
- **Output Layer**: Fixed to 2 neurons (for binary classification)
- **Output Activation**: Choose logistic or ReLU
- **Learning Rate**: Set learning rate (η) between 0.001 and 1.0
- **Beta**: Beta parameter for logistic activation (0.1-10.0)
- Click "🔧 Initialize Network"

**Important Notes:**
- For 2D input, the first hidden layer must have more than 2 neurons
- The output layer always has 2 neurons (one for each class)
- Total layers = Input (1) + Hidden (1-3) + Output (1) = 3-5 layers

#### 3. Train Network

- **Epochs**: Number of training epochs (1-1000)
- **Batch Size**: Batch size for training (1 = online learning, larger = batch training)
- **Reset Weights**: Option to reset weights before training
- **Variable Learning Rate**: Optional variable learning rate scheduling
  - If enabled, set η_min and η_max
- Click "🚀 Train Network"

#### 4. Visualize Results

**Visualization Tab:**
- For 2D data: Shows scatter plot with decision boundary (color-coded regions)
- For MNIST13: Shows sample images with predictions and confidence scores

**Training Error Tab:**
- Shows training error (MSE) over epochs
- Displays initial and final error values

## File Structure

```
4.5_Shallow_Neural_Network/
├── neural_network.py      # NeuralNetwork class with backpropagation
├── data_generator.py      # 2D data generator (from task 4.4)
├── mnist_loader.py        # MNIST13 dataset loader
├── gui.py                 # Streamlit GUI application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Grading Criteria

### Grade 3: Basic Implementation
- ✅ Evaluation and training (backpropagation) of 3-layered fully connected neural network
- ✅ Decision boundary visualization in GUI

### Grade 4: Parametrization
- ✅ Configurable number of layers (1-3 hidden layers = 3-5 total layers)
- ✅ Configurable number of neurons per layer

### Grade 5: Advanced Features
- ✅ Training on MNIST dataset
- ✅ Training in batches (configurable batch size)

## Technical Details

### Network Architecture
- **Input Layer**: 
  - 2 inputs for 2D generated data
  - 784 inputs for MNIST13 (28×28 flattened)
  - Multiple neurons in first hidden layer (>2 for 2D input)
- **Hidden Layers**: 1-3 layers, configurable neurons per layer
- **Output Layer**: 2 neurons (one per class)

### Activation Functions
- **Logistic (Sigmoid)**: `f(s) = 1 / (1 + exp(-β*s))`
  - Derivative: `f'(s) = β * f(s) * (1 - f(s))`
- **ReLU**: `f(s) = max(0, s)`
  - Derivative: `f'(s) = 1 if s > 0, else 0`

### Backpropagation Algorithm
The implementation follows the provided formula:
1. **Output layer**: `δ_k = (d_k - y_k) * f'(s_k)`
2. **Hidden layers**: `δ_k = Σ_i(δ_i * W_i^k) * f'(s_k)`
3. **Weight update**: `Δw_k^j = η * δ_k * x_l`

### Batch Training
- Supports full batch (batch_size = None)
- Supports mini-batch (batch_size = N, where N < dataset size)
- Supports online learning (batch_size = 1)
- Data is shuffled each epoch

## Troubleshooting

### MNIST Dataset Download Issues
- **Problem**: "Failed to load MNIST13"
- **Solution**: 
  - Ensure you have internet connection
  - Check that kagglehub is installed: `pip install kagglehub`
  - The dataset will be cached after first download

### Network Initialization Errors
- **Problem**: "For 2D input, first hidden layer must have more than 2 neurons"
- **Solution**: Set first hidden layer to have at least 3 neurons

- **Problem**: "Network can have at most 5 layers total"
- **Solution**: Reduce number of hidden layers (max 3 hidden layers)

### Training Issues
- **Problem**: Training error not decreasing
- **Solution**: 
  - Try adjusting learning rate
  - Try different activation functions
  - Increase number of epochs
  - For 2D data, try simpler network (1 hidden layer)

## Notes

- The decision boundary visualization works best for 2D data
- For MNIST13, the network operates on flattened 784-dimensional vectors
- The network outputs confidence values for each class (can be converted to probabilities)
- Training uses mean squared error (MSE) as the loss function
- Weights are initialized randomly in the range [-0.5, 0.5]

## References

- Task 4.4: Single Neuron Implementation (used as interface reference)
- Backpropagation formula from assignment specification
- MNIST dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
