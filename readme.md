# Scheem

Scheem is a lightweight, from-scratch neural network framework built entirely in Python from scratch (use numpy for fast calculation)

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

# Supported Activation Functions: sigmoid, relu, softmax
activation = ["sigmoid", "sigmoid"]

# Architecture: 2 inputs, 2 hidden neurons, 1 output
model = Scheem([2, 2, 1], activation, X, Y)
model.train(iters=10000, lr=1.0, log_after=1000)

# Iteration:    0 | Cost: 0.68658831
# Iteration: 1000 | Cost: 0.01567361
# Iteration: 2000 | Cost: 0.00599156
# Iteration: 3000 | Cost: 0.00367255
# Iteration: 4000 | Cost: 0.00264094
# Iteration: 5000 | Cost: 0.00205936
# Iteration: 6000 | Cost: 0.00168660
# Iteration: 7000 | Cost: 0.00142752
# Iteration: 8000 | Cost: 0.00123710
# Iteration: 9000 | Cost: 0.00109129

model.accuracy(X, Y) # generally X_test and Y_test 
# np.float64(1.0)

model.predict([0, 0])
# array([[0.00090511]])
```
- For more demos check out [demo directory](https://github.com/CYCNO/Scheem/tree/main/demos)
- For other implementation check out [other](https://github.com/CYCNO/Scheem/tree/main/other)

## Future Work
- [x] Build the Core Layers in Rust or C++ for Performance
- [x] Build in Numpy for simplicity and performance

## References:
- [Machine Learning In C | Tsoding](https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw)
- [Visualization Of Neural Network | 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [BackProp Notes](https://github.com/tsoding/ml-notes/blob/master/papers/grad.pdf)
