# Scheem

Scheem is a lightweight, from-scratch neural network framework built entirely in Python without using any external library

> Not to be Used in Production because it's very slow since it's written in pure python

## Quick Start

Here is how to use Scheem to solve the classic non-linear XOR Problem.

```python
from scheem import Scheem

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

# Architecture: 2 inputs, 2 hidden neurons, 1 output
# Supported Activation Functions: ["relu", "sigmoid"]
model = Scheem([2, 2, 1], X, Y, activation_function="sigmoid") 
model.fit(iter=10000, lr=1.0)

# iter = 0 | cost = 0.35620859892508483
# iter = 1000 | cost = 0.2492897995118044
# iter = 2000 | cost = 0.017144460455886114
# iter = 3000 | cost = 0.0028608591767603645
# iter = 4000 | cost = 0.0014556565141863624
# iter = 5000 | cost = 0.000960409604349115
# iter = 6000 | cost = 0.0007117208027857944
# iter = 7000 | cost = 0.0005632844823680261
# iter = 8000 | cost = 0.00046505465353611684
# iter = 9000 | cost = 0.0003954223205747279

# To check accuracy
model.accuracy(X, Y) # generally X_test and Y_test 
# 0.9815576780177311

# To predict
model.predict([0, 0])[0][0].data
# 0.982378009799738
```
- For more demos check out [demo directory](https://github.com/CYCNO/Scheem/tree/main/demos)
- For rust implementation check out [scheem_rust](https://github.com/CYCNO/Scheem/tree/main/scheem_rust)

## Architecture Under the Hood

Scheem is broken down into three core layers of abstraction:

#### 1. `Value` (The Autograd Engine)

At the lowest level, every number is wrapped in a `Value` object. When `.backward()` is called, it performs a topological sort and applies the chain rule to populate the `.grad` attribute of every node in the graph.

#### 2. `Matrix` (The Linear Algebra Layer)

Because neural networks require bulk calculations, the `Matrix` class organizes `Value` objects into 2D arrays. It handles matrix multiplication, random weight initialization (with both negative and positive bounds to break symmetry), and applies activation functions across all elements.

#### 3. `Scheem` (The High-Level API)

The `Scheem` class ties it all together. It manages the forward pass through the hidden layers, calculates the Mean Squared Error cost, and executes the training loop by updating weights using the calculated gradients.

## Future Work
- [x] Build the Core Layers in Rust or C++ for Performance

## References:
- [MicroGrad | Andrej Karpathy](https://github.com/karpathy/micrograd)
- [Machine Learning In C | Tsoding](https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw)
- [Visualization Of Neural Network | 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
