import numpy as np

class Scheem:
    def __init__(self, layers, activation, X_train, Y_train):
        self.X = np.array(X_train).T
        self.Y = np.array(Y_train).T
        self.m = self.X.shape[1]
        self.layers = layers
        self.n = len(layers)
        self.activation = activation

        self.W = []
        self.B = []
        self.Z = [None] * (self.n - 1)
        self.A = [None] * self.n

        # Initializing all the params
        for i in range(self.n - 1):
            scale = np.sqrt(2.0 / (layers[i] + layers[i + 1]))
            w = np.random.randn(layers[i + 1], layers[i]) * scale
            b = np.zeros((layers[i + 1], 1))
            self.W.append(w)
            self.B.append(b)

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500))) # to avoid large number error

    def sigmoid_deriv(self, A):
        return A * (1 - A)

    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self):
        self.A[0] = self.X
        for i in range(self.n - 1):
            self.Z[i] = self.W[i] @ self.A[i] + self.B[i]
            match self.activation[i]:
                case "relu":
                    self.A[i + 1] = self.ReLU(self.Z[i])
                case "sigmoid":
                    self.A[i + 1] = self.sigmoid(self.Z[i])
                case "softmax":
                    self.A[i + 1] = self.softmax(self.Z[i])

    def backprop(self):
        dW = [None] * (self.n - 1)
        dB = [None] * (self.n - 1)

        L = self.n - 2  # Output layer index

        # for output layer
        match self.activation[L]:
            case "sigmoid" | "softmax":
                # Natural pairing with Cross-Entropy: dZ simplifies to (A - Y)
                dZ = self.A[L + 1] - self.Y
            case "relu":
                # MSE pairing: dZ = 2 * (A - Y) * ReLU_deriv(Z)
                dZ = (self.A[L + 1] - self.Y) * self.ReLU_deriv(self.Z[L])

        dW[L] = (1 / self.m) * dZ @ self.A[L].T
        dB[L] = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)

        # for hidden layer
        for i in range(L - 1, -1, -1):
            match self.activation[i]:
                case "relu":
                    deriv = self.ReLU_deriv(self.Z[i])
                case "sigmoid":
                    deriv = self.sigmoid_deriv(self.A[i + 1])

            dZ = (self.W[i + 1].T @ dZ) * deriv
            dW[i] = (1 / self.m) * dZ @ self.A[i].T
            dB[i] = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)

        return dW, dB

    def cost(self):
        A = self.A[-1]

        match self.activation[-1]:
            case "sigmoid":
                # Binary Cross-Entropy
                A = np.clip(A, 1e-15, 1 - 1e-15)
                return -np.mean(self.Y * np.log(A) + (1 - self.Y) * np.log(1 - A))
            case "softmax":
                # Categorical Cross-Entropy
                A = np.clip(A, 1e-15, 1 - 1e-15)
                return -np.mean(np.sum(self.Y * np.log(A), axis=0))
            case "relu":
                # Mean Squared Error (MSE) for regression / continuous non-binary values
                return np.mean((A - self.Y) ** 2)

    def train(self, iters=1000, lr=0.1, log_after=100):
        for i in range(iters):
            self.forward()
            dW, dB = self.backprop()

            for k in range(self.n - 1):
                self.W[k] -= lr * dW[k]
                self.B[k] -= lr * dB[k]

            if i % log_after == 0:
                print(f"Iteration: {i:4d} | Cost: {self.cost():.8f}")

    def predict(self, X):
        X_arr = np.atleast_2d(X)
        A = [None] * self.n
        A[0] = X_arr.T

        for i in range(self.n - 1):
            Z = self.W[i] @ A[i] + self.B[i]
            match self.activation[i]:
                case "relu":
                    A[i + 1] = self.ReLU(Z)
                case "sigmoid":
                    A[i + 1] = self.sigmoid(Z)
                case "softmax":
                    A[i + 1] = self.softmax(Z)
        return A[-1]

    def accuracy(self, X_test, Y_test):
        pred = self.predict(X_test)
        Y = np.array(Y_test).T

        match self.activation[-1]:
            case "sigmoid":
                # Binary classification accuracy
                return np.mean(np.round(pred) == Y)
            case "softmax":
                # Multi-class accuracy
                return np.mean(np.argmax(pred, axis=0) == np.argmax(Y, axis=0))
            case "relu":
                # Regression evaluation: R^2 Score
                ss_res = np.sum((Y - pred) ** 2)
                ss_tot = np.sum((Y - np.mean(Y)) ** 2)
                return 1 - (ss_res / (ss_tot + 1e-12))
