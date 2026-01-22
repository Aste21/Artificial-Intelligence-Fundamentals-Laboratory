# Convolutional Neural Network Implementation

Implementation of a Convolutional Neural Network (CNN) with custom convolution operator, convolution layers, and max-pooling layers for MNIST digit classification.

## Features

### 1. Custom Convolution Operator (Grade 3)
- **`convolve_2d()`**: Custom implementation of 2D convolution for greyscale images
- Supports stride and padding
- Validated with provided example

### 2. Simple CNN (Grade 4)
- **Convolution Layer**: Uses NumPy's `scipy.signal.convolve2d` for convolution
- **ReLU Activation**: Applied after convolution
- **Fully Connected Layer**: Compatible with task 4.5 architecture
- **Training**: Trained on MNIST dataset with 2D images

### 3. CNN with Pooling (Grade 5)
- **Max-Pooling Layer**: Configurable pooling size and stride
- **Complete Architecture**: Conv → ReLU → Max-Pool → FC → Softmax
- **Training**: Full training pipeline on MNIST

## Installation

### Prerequisites
- Python 3.7 or higher
- Internet connection (for downloading MNIST dataset on first run)

### Quick Start (Windows CMD)

```cmd
cd 4.6_Convolutional_Neural_Network
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python train_cnn.py
```

### Quick Start (Windows PowerShell)

```powershell
cd 4.6_Convolutional_Neural_Network
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_cnn.py
```

**Szczegółowe instrukcje:** Zobacz plik `SETUP.md` w folderze projektu.

## Usage

### Test Custom Convolution Operator

```bash
python convolution.py
```

This will test the convolution operator with the provided example:
- Input: 5x5 image
- Filter: 3x3 kernel
- Expected output: 3x3 result

### Train Simple CNN

```bash
python train_cnn.py
```

This will:
1. Load MNIST dataset (downloads automatically on first run)
2. Train a simple CNN: Conv(3x3, 16 filters) → ReLU → FC(10 neurons)
3. Train a CNN with pooling: Conv(3x3, 16 filters) → ReLU → MaxPool(2x2) → FC(10 neurons) → Softmax
4. Evaluate and report accuracy

## Architecture Details

### Simple CNN
```
Input: (28, 28) greyscale image
  ↓
Conv: 3x3, 16 filters, stride=1, padding=0
  ↓
ReLU activation
  ↓
Flatten
  ↓
Fully Connected: 10 neurons
  ↓
Output: 10 class probabilities
```

### CNN with Pooling
```
Input: (28, 28) greyscale image
  ↓
Conv: 3x3, 16 filters, stride=1, padding=0
  ↓
ReLU activation
  ↓
Max-Pooling: 2x2, stride=2
  ↓
Flatten
  ↓
Fully Connected: 10 neurons
  ↓
Softmax activation
  ↓
Output: 10 class probabilities
```

## File Structure

```
4.6_Convolutional_Neural_Network/
├── convolution.py          # Custom 2D convolution operator
├── cnn_layers.py           # Convolution and pooling layers
├── cnn.py                  # CNN implementation
├── mnist_loader_2d.py      # MNIST loader (2D images)
├── train_cnn.py            # Training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Implementation Details

### Custom Convolution Operator
- Implemented from scratch using nested loops
- Supports arbitrary filter sizes, strides, and padding
- Validated against provided example

### Convolution Layer
- Uses `scipy.signal.convolve2d` for efficient convolution
- Supports multiple filters and input channels
- Handles stride by subsampling
- Compatible with task 4.5 layer interface

### Max-Pooling Layer
- Configurable pooling window size
- Configurable stride
- Tracks max indices for potential backpropagation

### Training
- Forward pass through all layers
- Backpropagation through fully connected layers
- Cross-entropy loss with softmax
- Batch training support

## Grading Criteria

### Grade 3: Custom Convolution Operator
- ✅ Implementation of `convolve_2d()` for 2D greyscale images
- ✅ Supports stride and padding
- ✅ Validated with provided example

### Grade 4: Simple CNN
- ✅ Convolution layer using NumPy's convolution
- ✅ Simple CNN: Conv(3x3, 16 filters) → ReLU → FC(10 neurons)
- ✅ Training on MNIST dataset (2D images)

### Grade 5: CNN with Pooling
- ✅ Max-pooling layer with configurable size and stride
- ✅ Complete CNN: Conv → ReLU → MaxPool → FC → Softmax
- ✅ Training on MNIST dataset

## Notes

- The MNIST dataset is automatically downloaded on first run using `kagglehub`
- Images are provided as 2D tensors (28x28), not flattened
- Training uses a subset of data by default for faster execution
- Full backpropagation through convolutional layers is simplified (focuses on FC layers)
- For production use, consider implementing full backpropagation through conv layers

## Troubleshooting

### MNIST Download Issues
- **Problem**: "Failed to download MNIST dataset"
- **Solution**: Ensure internet connection and `kagglehub` is installed

### Import Errors
- **Problem**: "No module named 'scipy'"
- **Solution**: Install dependencies: `pip install -r requirements.txt`

### Memory Issues
- **Problem**: Out of memory during training
- **Solution**: Reduce batch size in `train_cnn.py` or use fewer samples

## References

- Task 4.5: Shallow Neural Network (used as architecture reference)
- MNIST dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- NumPy convolution: `scipy.signal.convolve2d`
