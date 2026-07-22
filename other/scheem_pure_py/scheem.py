from matrix import Matrix

class Scheem:
    def __init__(self, layers, X, Y, activation_function="relu"):
        self.X = Matrix(X)  # For inputs
        self.Y = Matrix(Y)  # For Outputs

        self.activation_function = activation_function

        self.W = []
        self.B = []

        for i in range(0, len(layers) - 1):
            # input = layers[i], output = layers[i+1]
            w = Matrix(layers[i], layers[i + 1])
            b = Matrix(1, layers[i + 1])

            w.random(-1, 1)
            b.random(-1, 1)

            self.W.append(w)
            self.B.append(b)

    def forward(self, x_batch):
        activation = x_batch if isinstance(x_batch, Matrix) else Matrix(x_batch)
        if self.activation_function not in ["relu", "sigmoid"]:
            raise NotImplementedError(f"Activation function '{self.activation_function}' is not supported.")

        for i in range(len(self.W)):
            z = (activation * self.W[i]) + self.B[i]

            if i < len(self.W) - 1:
                if self.activation_function == "relu":
                    activation = Matrix.relu(z)
                elif self.activation_function == "sigmoid":
                    activation = Matrix.sigmoid(z)
            else: # last layer
                if self.activation_function == "relu":
                    activation = z
                elif self.activation_function == "sigmoid":
                    activation = Matrix.sigmoid(z)

        return activation


    def cost(self):
        if self.X.rows != self.Y.rows:
            print("Rows must be same of both inputs and outputs")
            return

        cost = 0
        for i in range(self.X.rows):
            activation = self.forward(self.X[i])
            for j in range(self.Y.cols):
                d = self.Y[i][j] - activation[0][j]

                cost += d * d
        return cost / (self.X.rows)

    def fit(self, iter=10000, lr=0.01):
        params = [w for rows in self.W for l in rows for w in l] + [
            b for rows in self.B for l in rows for b in l
        ]

        for i in range(iter):
            # zero grad all the params
            for p in params:
                p.grad = 0.0

            # calculate loss and backprop
            loss = self.cost()
            loss.backward()

            # update the params
            for p in params:
                p.data -= lr * p.grad

            if i % 1000 == 0:
                print(f"iter = {i} | cost = {loss}")


    def accuracy(self, X_test, Y_test):
        error = 0
        for i in range(len(X_test)):
            x_batch = X_test[i]
            X = x_batch if isinstance(x_batch, Matrix) else Matrix(x_batch)
            pred = self.forward(x_batch)
            error += abs(Y_test[i][0] - pred[0][0].data)
        return 1 - (error / len(X_test))

    def predict(self, X):
        X = X if isinstance(X, Matrix) else Matrix(X)
        z = self.forward(X)
        return z


# TODO: Make sigmoid and ReLU different
