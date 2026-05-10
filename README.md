# Handwritten-Digit-Identifier-Neural-Network

A handwritten digit recognition neural network built completely from scratch in Python using NumPy.

This project implements:

- Feedforward neural network
- Backpropagation
- Mini-batch stochastic gradient descent (SGD)
- Sigmoid activation
- MNIST handwritten digit recognition

The network is trained on the MNIST dataset and achieves around 94–95% accuracy.

---

# Network Architecture

784 → 30 → 10

- 784 Input Neurons (28×28 image pixels)
- 30 Hidden Neurons
- 10 Output Neurons (digits 0–9)

---

# Features

- Neural network implemented completely from scratch
- No machine learning frameworks used
- Matrix-based backpropagation
- MNIST dataset support
- Mini-batch SGD training
- Python 3 compatible

---

# Project Structure

```text
Digit_Recognizer_Neural_Network/
│
├── network.py
├── mnist_loader.py
├── main.py
├── requirements.txt
├── README.md
│
└── data/
    └── mnist.pkl.gz
```

---

# Installation

## Clone Repository

```bash
git clone <your-repository-link>
cd Digit_Recognizer_Neural_Network
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Download MNIST Dataset

Download:

https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz

Place it inside:

```text
data/
```

---

# Run The Project

```bash
python main.py
```

---

# Example Output

```text
Epoch 0: 9065 / 10000
Epoch 1: 9189 / 10000
Epoch 2: 9255 / 10000
...
```

---

# How It Works

The neural network learns handwritten digits using:

1. Feedforward propagation
2. Cost calculation
3. Backpropagation
4. Gradient descent updates

The model uses sigmoid activation functions and learns by adjusting weights and biases over multiple epochs.

---

# Technologies Used

- Python 3
- NumPy
- Matplotlib

---

# Future Improvements

- ReLU activation
- Cross-entropy loss
- Better weight initialization
- CNN implementation
- Neural network visualization
- Real-time training visualization

---

# Acknowledgements

Inspired by:

"Neural Networks and Deep Learning" by Michael Nielsen

https://github.com/mnielsen/neural-networks-and-deep-learning
