# Handwritten Digit Recognition — From Scratch

A neural network built entirely from scratch with NumPy (no TensorFlow/PyTorch) that
recognizes hand-drawn digits (0-9) from the MNIST dataset, achieving **~92.5% validation accuracy**.

This is an update to a previous version of this project that used TensorFlow. This time,
forward propagation, backpropagation, and gradient descent are all implemented manually
to better understand how a neural network actually works under the hood.

## Architecture

- Input layer: 784 units (28x28 flattened pixel values)
- Hidden layer: 10 units, ReLU activation
- Output layer: 10 units, Softmax activation
- Trained with plain gradient descent (no optimizer libraries)

## Files

- `main.ipynb` — data loading, model definition, training, and evaluation
- `draw_digit.py` — a Tkinter app to draw a digit and get a live prediction from the trained model
- `W1.txt`, `b1.txt`, `W2.txt`, `b2.txt` — saved weights/biases from training

## Usage

1. Download the [MNIST CSV dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and place `train.csv` (and optionally `test.csv`) in the project root.
2. Run `main.ipynb` to train the model and save the weights.
3. Run `draw_digit.py` to draw a digit and see the model predict it in real time:

```bash
python draw_digit.py
```
