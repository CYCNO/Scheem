# Scheem

**Scheem** is a lightweight, from-scratch neural network framework built entirely in Python without using any external library

## Quick Start

Here is how to use Scheem to solve the classic non-linear **XOR Problem**.

```python
from scheem import Scheem

# XOR dataset
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y = [
    [0],
    [1],
    [1],
    [0]
]

# Architecture: 2 inputs, 4 hidden neurons, 1 output
model = Scheem(layers=[2, 4, 1], X=X, Y=Y, activation_function="sigmoid") # supported: ["relu", "sigmoid"]

model.fit(iter=20000, lr=1.0)
```

## Architecture Under the Hood

Scheem is broken down into three core layers of abstraction:

### 1. `Value` (The Autograd Engine) (Inspired by MicroGrad)

At the lowest level, every number is wrapped in a `Value` object. This class implements Python magic methods (`__add__`, `__mul__`, etc.) to track operations. When `.backward()` is called, it performs a topological sort and applies the chain rule to populate the `.grad` attribute of every node in the graph.

### 2. `Matrix` (The Linear Algebra Layer)

Because neural networks require bulk calculations, the `Matrix` class organizes `Value` objects into 2D arrays. It handles matrix multiplication, random weight initialization (with both negative and positive bounds to break symmetry), and applies activation functions across all elements.

### 3. `Scheem` (The High-Level API)

The `Scheem` class ties it all together. It manages the forward pass through the hidden layers, calculates the Mean Squared Error cost, and executes the training loop by updating weights using the calculated gradients.

## References:
